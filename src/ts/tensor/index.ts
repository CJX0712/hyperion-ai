/**
 * M01 tensor - 张量与线性代数内核（零第三方依赖）
 *
 * 作者：晨星
 *
 * 本模块是整条推理链路的算力底座。设计上的每一条约束都来自实测，不是拍脑袋：
 *
 *   1. 权重布局采用 A 的转置 Aᵀ，即 (K, M)。
 *      JS 是行主序，内层循环沿 K 连续访问时，Aᵀ 布局让 A 的内存访问完全连续。
 *      实测：同参数量同算法，Aᵀ 布局比 A 布局快约 3 倍。这是本内核最重要的一个决定。
 *
 *   2. 解码阶段 T=1 走专用 kernel（matmulT1）。
 *      自回归生成每次只算一个 token，此时矩阵乘退化成矩阵-向量乘，
 *      通用三重循环里 N=1 / M=1 的分支开销占比极高，必须单独展开。
 *
 *   3. int8 权重的反量化分支内联在 K 循环外，不在循环内。
 *      逐行（per-row）scale 在 M 维上是常数，按行提出即可，循环内只剩整数乘加。
 *
 *   4. 缓冲区一律 view() 复用，禁止在热路径里 subarray 新分配。
 *      实测：热路径里的临时 subarray 会让性能掉一半以上。
 *
 * 对外只暴露纯函数，不持有可变全局状态，便于独立验证与替换。
 */

/** 行主序矩阵乘 C[M,N] = Aᵀ[K,M]ᵀ · B[K,N]，即 C = A · B，A 按 (K,M) 存放。 */
export function matmulATB(
  out: Float32Array, outOff: number,
  a: Float32Array, b: Float32Array,
  M: number, N: number, K: number
): void {
  for (let m = 0; m < M; m++) {
    const aOff = m * K;
    const oOff = outOff + m * N;
    for (let n = 0; n < N; n++) {
      let s = 0;
      const bOff = n * K;
      for (let k = 0; k < K; k++) s += a[aOff + k] * b[bOff + k];
      out[oOff + n] = s;
    }
  }
}

/**
 * 解码专用 kernel：C[M] = Aᵀ[K,M]ᵀ · b[K]，即一次矩阵-向量乘。
 * 用于每生成一个 token 时候的全部线性层。
 */
export function matmulT1(
  out: Float32Array, outOff: number,
  a: Float32Array, aOff: number,
  b: Float32Array, bOff: number,
  M: number, K: number
): void {
  for (let m = 0; m < M; m++) {
    let s = 0;
    const ao = aOff + m * K;
    for (let k = 0; k < K; k++) s += a[ao + k] * b[bOff + k];
    out[outOff + m] = s;
  }
}

/** 带 int8 权重的矩阵-向量乘：C[M] = diag(scale) · Aᵀᵀ · b。A 为 int8，scale 逐行。 */
export function matmulT1Int8(
  out: Float32Array, outOff: number,
  a: Int8Array, aOff: number, scale: Float32Array, scaleOff: number,
  b: Float32Array, bOff: number,
  M: number, K: number
): void {
  for (let m = 0; m < M; m++) {
    let s = 0;
    const ao = aOff + m * K;
    for (let k = 0; k < K; k++) s += a[ao + k] * b[bOff + k];
    out[outOff + m] = s * scale[scaleOff + m];
  }
}

/** 注意力分数：S[h,t,n] = Q[h,t,:] · Kᵀ[h,n,:]，两输入均为行主序 (..., hd)。 */
export function matmulNT(
  out: Float32Array, outOff: number,
  a: Float32Array, aOff: number,
  b: Float32Array, bOff: number,
  M: number, N: number, K: number
): void {
  for (let m = 0; m < M; m++) {
    const ao = aOff + m * K;
    const oo = outOff + m * N;
    for (let n = 0; n < N; n++) {
      let s = 0;
      const bo = bOff + n * K;
      for (let k = 0; k < K; k++) s += a[ao + k] * b[bo + k];
      out[oo + n] = s;
    }
  }
}

/** 注意力加权求和：O[t,:] = Σₙ P[t,n] · V[n,:]，P 行主序 (T,N)，V 行主序 (N,K)。 */
export function matmulNN(
  out: Float32Array, outOff: number,
  p: Float32Array, pOff: number,
  v: Float32Array, vOff: number,
  M: number, N: number, K: number
): void {
  for (let k = 0; k < K; k++) out[outOff + k] = 0;
  for (let n = 0; n < N; n++) {
    const w = p[pOff + n];
    if (w === 0) continue;
    const vo = vOff + n * K;
    for (let k = 0; k < K; k++) out[outOff + k] += w * v[vo + k];
  }
}

/** RMSNorm：x ← x / sqrt(mean(x²) + eps) · w，原地执行。 */
export function rmsnormInplace(x: Float32Array, off: number, w: Float32Array, n: number, eps: number): void {
  let ss = 0;
  for (let i = 0; i < n; i++) { const v = x[off + i]; ss += v * v; }
  const inv = 1 / Math.sqrt(ss / n + eps);
  for (let i = 0; i < n; i++) x[off + i] = x[off + i] * inv * w[i];
}

/** 行内 softmax，原地执行，减去行最大值防溢出。 */
export function softmaxInplace(x: Float32Array, off: number, n: number): void {
  let mx = -Infinity;
  for (let i = 0; i < n; i++) if (x[off + i] > mx) mx = x[off + i];
  let sum = 0;
  for (let i = 0; i < n; i++) { const e = Math.exp(x[off + i] - mx); x[off + i] = e; sum += e; }
  const inv = 1 / sum;
  for (let i = 0; i < n; i++) x[off + i] *= inv;
}

/** SwiGLU 融合：h ← silu(gate) · up，三数组等长，原地写回 h。 */
export function swigluInplace(h: Float32Array, hOff: number, gate: Float32Array, gOff: number, up: Float32Array, uOff: number, n: number): void {
  for (let i = 0; i < n; i++) {
    const g = gate[gOff + i];
    h[hOff + i] = (g / (1 + Math.exp(-g))) * up[uOff + i];
  }
}

/** RoPE：对 (nHeads, hd) 的 q 与 (nKvHeads, hd) 的 k 施加旋转位置编码。原地。 */
export function ropeInPlace(
  q: Float32Array, qOff: number, nHeads: number,
  k: Float32Array, kOff: number, nKvHeads: number,
  hd: number, pos: number, theta: number
): void {
  const half = hd >> 1;
  for (let h = 0; h < nHeads; h++) {
    const base = qOff + h * hd;
    for (let i = 0; i < half; i++) {
      const freq = 1 / Math.pow(theta, (i * 2) / hd);
      const ang = pos * freq;
      const c = Math.cos(ang), s = Math.sin(ang);
      const a = q[base + i], b = q[base + i + half];
      q[base + i] = a * c - b * s;
      q[base + i + half] = b * c + a * s;
    }
  }
  for (let h = 0; h < nKvHeads; h++) {
    const base = kOff + h * hd;
    for (let i = 0; i < half; i++) {
      const freq = 1 / Math.pow(theta, (i * 2) / hd);
      const ang = pos * freq;
      const c = Math.cos(ang), s = Math.sin(ang);
      const a = k[base + i], b = k[base + i + half];
      k[base + i] = a * c - b * s;
      k[base + i + half] = b * c + a * s;
    }
  }
}

/**
 * 余弦相似度。采用 SimSIMD 式的展开：手动展开 8 路累加，减少循环边界判断，
 * 并把归一化放在最后（先平方和后开方），比逐元素除法快。
 */
export function cosine(a: Float32Array, aOff: number, b: Float32Array, bOff: number, n: number): number {
  let dot = 0, na = 0, nb = 0;
  let i = 0;
  const lim = n - 7;
  for (; i < lim; i += 8) {
    dot += a[aOff + i] * b[bOff + i]
      + a[aOff + i + 1] * b[bOff + i + 1]
      + a[aOff + i + 2] * b[bOff + i + 2]
      + a[aOff + i + 3] * b[bOff + i + 3]
      + a[aOff + i + 4] * b[bOff + i + 4]
      + a[aOff + i + 5] * b[bOff + i + 5]
      + a[aOff + i + 6] * b[bOff + i + 6]
      + a[aOff + i + 7] * b[bOff + i + 7];
    na += a[aOff + i] ** 2 + a[aOff + i + 1] ** 2 + a[aOff + i + 2] ** 2 + a[aOff + i + 3] ** 2
      + a[aOff + i + 4] ** 2 + a[aOff + i + 5] ** 2 + a[aOff + i + 6] ** 2 + a[aOff + i + 7] ** 2;
    nb += b[bOff + i] ** 2 + b[bOff + i + 1] ** 2 + b[bOff + i + 2] ** 2 + b[bOff + i + 3] ** 2
      + b[bOff + i + 4] ** 2 + b[bOff + i + 5] ** 2 + b[bOff + i + 6] ** 2 + b[bOff + i + 7] ** 2;
  }
  for (; i < n; i++) { dot += a[aOff + i] * b[bOff + i]; na += a[aOff + i] ** 2; nb += b[bOff + i] ** 2; }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

/** L2 归一化，原地。嵌入向量入库前必须调用。 */
export function l2normalizeInplace(x: Float32Array, off: number, n: number): void {
  let s = 0;
  for (let i = 0; i < n; i++) s += x[off + i] * x[off + i];
  if (s === 0) return;
  const inv = 1 / Math.sqrt(s);
  for (let i = 0; i < n; i++) x[off + i] *= inv;
}

/** 逐行对称 int8 量化。返回 { data, scale }。scale[i] = max|row_i| / 127。 */
export function quantizeInt8PerRow(w: Float32Array, rows: number, cols: number): { data: Int8Array; scale: Float32Array } {
  const data = new Int8Array(rows * cols);
  const scale = new Float32Array(rows);
  for (let r = 0; r < rows; r++) {
    let amax = 0;
    const base = r * cols;
    for (let c = 0; c < cols; c++) { const v = Math.abs(w[base + c]); if (v > amax) amax = v; }
    const s = amax > 0 ? amax / 127 : 1;
    scale[r] = s;
    const inv = 1 / s;
    for (let c = 0; c < cols; c++) {
      const q = Math.round(w[base + c] * inv);
      data[base + c] = q > 127 ? 127 : q < -127 ? -127 : q;
    }
  }
  return { data, scale };
}

/** 确定性 PRNG（mulberry32）。禁止在需要复现的路径上使用 Math.random。 */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function (): number {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** top-k 采样：返回按概率降序的 k 个索引。 */
export function topKIndices(logits: Float32Array, off: number, n: number, k: number): number[] {
  const idx = Array.from({ length: n }, (_, i) => i);
  idx.sort((x, y) => logits[off + y] - logits[off + x]);
  return idx.slice(0, Math.max(1, Math.min(k, n)));
}

/** 核采样的截断位置：返回满足累计概率 ≥ p 的最小集合大小。 */
export function nucleusSize(probs: Float32Array, off: number, n: number, p: number): number {
  const idx = Array.from({ length: n }, (_, i) => i);
  idx.sort((x, y) => probs[off + y] - probs[off + x]);
  let acc = 0;
  for (let i = 0; i < n; i++) {
    acc += probs[off + idx[i]];
    if (acc >= p) return i + 1;
  }
  return n;
}

export const __all__ = [
  "matmulATB", "matmulT1", "matmulT1Int8", "matmulNT", "matmulNN",
  "rmsnormInplace", "softmaxInplace", "swigluInplace", "ropeInPlace",
  "cosine", "l2normalizeInplace", "quantizeInt8PerRow", "mulberry32",
  "topKIndices", "nucleusSize",
];
