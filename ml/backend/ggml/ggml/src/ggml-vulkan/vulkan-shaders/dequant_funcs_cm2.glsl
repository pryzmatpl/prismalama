
#include "types.glsl"

layout(buffer_reference, std430, buffer_reference_align = 16) buffer decodeBufF32 {
   vec4 block;
};

float16_t dequantFuncF32(const in decodeBufF32 bl, const in uint blockCoords[2], const in uint coordInBlock[2])
{
    const vec4 v = bl.block;
    const uint idx = coordInBlock[1];
    const f16vec4 vf16 = f16vec4(v);
    return vf16[idx];
}

layout(buffer_reference, std430, buffer_reference_align = 2) buffer decodeBufQ4_0 {
   block_q4_0_packed16 block;
};

float16_t dequantFuncQ4_0(const in decodeBufQ4_0 bl, const in uint blockCoords[2], const in uint coordInBlock[2])
{
    const float16_t d = bl.block.d;
    const uint idx = coordInBlock[1];
    const uint shift = (idx & 0x10) >> 2;
    uint32_t qs = uint32_t(bl.block.qs[(idx & 0xE) >> 1]);
    qs >>= shift;
    qs &= 0x0F0F;
    qs = unpack8(qs)[idx & 1];
    float16_t ret = (float16_t(qs) - float16_t(8)) * d;
    return ret;
}

layout(buffer_reference, std430, buffer_reference_align = 4) buffer decodeBufQ4_1 {
   block_q4_1 block;
};

float16_t dequantFuncQ4_1(const in decodeBufQ4_1 bl, const in uint blockCoords[2], const in uint coordInBlock[2])
{
    const float16_t d = bl.block.d;
    const float16_t m = bl.block.m;
    const uint idx = coordInBlock[1];
    const uint iqs = idx & 0xF;
    const uint shift = (idx & 0x10) >> 2;
    uint32_t qs = bl.block.qs[iqs];
    qs >>= shift;
    qs &= 0xF;
    float16_t ret = float16_t(qs) * d + m;
    return ret;
}

layout(buffer_reference, std430, buffer_reference_align = 2) buffer decodeBufQ5_0 {
   block_q5_0 block;
};

float16_t dequantFuncQ5_0(const in decodeBufQ5_0 bl, const in uint blockCoords[2], const in uint coordInBlock[2])
{
    const float16_t d = bl.block.d;
    const uint idx = coordInBlock[1];
    const uint iqs = idx & 0xF;

    const uint uint_qh = uint(bl.block.qh[1]) << 16 | bl.block.qh[0];
    const uint qh = ((uint_qh >> idx) << 4) & 0x10;

    const uint shift = (idx & 0x10) >> 2;
    uint32_t qs = bl.block.qs[iqs];
    qs >>= shift;
    qs &= 0xF;

    float16_t ret = (float16_t(qs | qh) - float16_t(16)) * d;
    return ret;
}

layout(buffer_reference, std430, buffer_reference_align = 8) buffer decodeBufQ5_1 {
   block_q5_1 block;
};

float16_t dequantFuncQ5_1(const in decodeBufQ5_1 bl, const in uint blockCoords[2], const in uint coordInBlock[2])
{
    const float16_t d = bl.block.d;
    const float16_t m = bl.block.m;
    const uint idx = coordInBlock[1];
    const uint iqs = idx & 0xF;

    const uint uint_qh = bl.block.qh;
    const uint qh = ((uint_qh >> idx) << 4) & 0x10;

    const uint shift = (idx & 0x10) >> 2;
    uint32_t qs = bl.block.qs[iqs];
    qs >>= shift;
    qs &= 0xF;

    float16_t ret = float16_t(qs | qh) * d + m;
    return ret;
}

layout(buffer_reference, std430, buffer_reference_align = 2) buffer decodeBufQ8_0 {
   block_q8_0_packed16 block;
};

float16_t dequantFuncQ8_0(const in decodeBufQ8_0 bl, const in uint blockCoords[2], const in uint coordInBlock[2])
{
    const float16_t d = bl.block.d;
    const uint idx = coordInBlock[1];
    const uint iqs = idx;

    // Load 16b and select the byte for this element
    int32_t qs = unpack8(bl.block.qs[(iqs & 0x1E) >> 1])[iqs & 1];
    float16_t ret = float16_t(qs) * d;
    return ret;
}

layout(buffer_reference, std430, buffer_reference_align = 4) buffer decodeBufQ2_K {
   block_q2_K block;
};

layout(buffer_reference, std430, buffer_reference_align = 16) buffer decodeBufQ2_K_packed16 {
   block_q2_K_packed16 block;
};

float16_t dequantFuncQ2_K(const in decodeBufQ2_K bl, const in uint blockCoords[2], const in uint coordInBlock[2])
{
    decodeBufQ2_K_packed16 bl16 = decodeBufQ2_K_packed16(bl);
    const f16vec2 dm = bl.block.dm;
    const uint idx = coordInBlock[1];

    const uint scalesi = (idx & 0xF0) >> 4;             // 0..15
    const uint qsshift = (idx & 0x60) >> 4;             // 0,2,4,6

    uint qs = uint32_t(bl16.block.qs[((idx & 0x80) >> 3) + ((idx & 0x1E) >> 1)]);
    qs = (qs >> qsshift) & 0x0303;
    qs = unpack8(qs)[idx & 1];

    const uint scales = bl.block.scales[scalesi];
    float16_t ret = dm.x * float16_t(scales & 0xF) * float16_t(qs) - dm.y * float16_t(scales >> 4);
    return ret;
}

layout(buffer_reference, std430, buffer_reference_align = 2) buffer decodeBufQ3_K {
   block_q3_K block;
};

float16_t dequantFuncQ3_K(const in decodeBufQ3_K bl, const in uint blockCoords[2], const in uint coordInBlock[2])
{
    const uint idx = coordInBlock[1];
    const uint iqs = idx;

    const uint n = iqs / 128;                    // 0,1
    const uint qsi = n * 32 + (iqs % 32);        // 0..63
    const uint hmi =          (iqs % 32);        // 0..31
    const uint j = (iqs % 128) / 8;              // 0..15
    const uint is = iqs / 16;                    // 0..15
    const uint halfsplit = ((iqs % 128) / 32);   // 0,1,2,3
    const uint qsshift = halfsplit * 2;          // 0,2,4,6
    const uint m = 1 << (4 * n + halfsplit);     // 1,2,4,8,16,32,64,128

    uint32_t scaleidx0 = (is < 8) ? is : (is-8);
    uint32_t scaleidx0shift = (is < 8) ? 0 : 4;
    uint32_t scaleidx1 = is + 8 - (is/4)*4;
    uint32_t scaleidx1shift = (is/4)*2;

    const int8_t us = int8_t(((bl.block.scales[scaleidx0] >> scaleidx0shift) & 0xF) | (((bl.block.scales[scaleidx1] >> scaleidx1shift) & 3) << 4));

    const float16_t dl = bl.block.d * float16_t(us - 32);

    float16_t ret = dl * float16_t(int8_t((bl.block.qs[qsi    ] >> qsshift) & 3) - (((bl.block.hmask[hmi    ] & m) != 0) ? 0 : 4));

    return ret;
}

layout(buffer_reference, std430, buffer_reference_align = 16) buffer decodeBufQ4_K {
   block_q4_K block;
};

layout(buffer_reference, std430, buffer_reference_align = 16) buffer decodeBufQ4_K_packed16 {
   block_q4_K_packed16 block;
};

layout(buffer_reference, std430, buffer_reference_align = 16) buffer decodeBufQ4_K_packed128 {
   block_q4_K_packed128 block;
};

#if defined(IS_MUL_MM2)

// For Q4_K and Q5_K in the mat-mul shader, we decode a tile's worth of scales
// into shared memory and then process the whole tile using those scales.
// There is a fetch function that loads into private variables and then a store
// function that stores into shared memory.
// Q4_K and Q5_K have the same encoding of scales, so everything is shared except
// the part that fetches from the structure (which has a different block layout).
#if defined(DATA_A_Q4_K) || defined(DATA_A_Q5_K)
const uint shAscales_stride = (BM + 2);
// 1 scale per 32 elements -> 8 scales per block, per row
shared vec2 shAscales[8 * shAscales_stride];
uvec4 row_v;
#endif

#if defined(DATA_A_Q4_K)
layout (binding = 0) readonly buffer A_Q4_K_128 {block_q4_K_packed128 data_a_q4_k_packed128[];};

void fetch_scalesQ4_K(uint ir_BM, uint pos_a, uint stride_a, uint block_k, uint tid, bool in_bounds)
{
    uint tids_per_row = BLOCK_SIZE / BM;
    uint is_per_tid = 8 / tids_per_row;
    uint is_start = is_per_tid * (tid % tids_per_row);
    uint tid_row = tid / tids_per_row;

    uint row = ir_BM + tid_row;
    uint block_index = pos_a + row * stride_a + (block_k / QUANT_K);
    if (in_bounds || row < p.M) {
        row_v = data_a_q4_k_packed128[block_index].q4k[0];
    }
}
#endif
#if defined(DATA_A_Q5_K)
layout (binding = 0) readonly buffer A_Q5_K_128 {block_q5_K_packed128 data_a_q5_k_packed128[];};

void fetch_scalesQ5_K(uint ir_BM, uint pos_a, uint stride_a, uint block_k, uint tid, bool in_bounds)
{
    uint tids_per_row = BLOCK_SIZE / BM;
    uint is_per_tid = 8 / tids_per_row;
    uint is_start = is_per_tid * (tid % tids_per_row);
    uint tid_row = tid / tids_per_row;

    uint row = ir_BM + tid_row;
    uint block_index = pos_a + row * stride_a + (block_k / QUANT_K);
    if (in_bounds || row < p.M) {
        row_v = data_a_q5_k_packed128[block_index].q5k[0];
    }
}
#endif

#if defined(DATA_A_Q4_K) || defined(DATA_A_Q5_K)
void store_scalesQ4_K(uint tid)
{
    barrier();

    uint tids_per_row = BLOCK_SIZE / BM;
    uint is_per_tid = 8 / tids_per_row;
    uint is_start = is_per_tid * (tid % tids_per_row);
    uint tid_row = tid / tids_per_row;

    [[unroll]] for (uint idx = 0; idx < is_per_tid; ++idx) {
        uint is = idx + is_start;
        uvec4 v = row_v;
        const vec2 loadd = vec2(unpackFloat2x16(v.x));

        uint32_t sc;
        uint32_t mbyte;

        uint32_t scale0 = v.y;
        uint32_t scale4 = v.z;
        uint32_t scale8 = v.w;

        uint32_t sc_lo = scale0;
        uint32_t mb_lo = scale4;
        uint32_t sc_hi = (scale8 & 0x0F0F0F0F) | ((scale0 & 0xC0C0C0C0) >> 2);
        uint32_t mb_hi = ((scale8 & 0xF0F0F0F0) >> 4) | ((scale4 & 0xC0C0C0C0) >> 2);

        sc = is < 4 ? sc_lo : sc_hi;
        mbyte = is < 4 ? mb_lo : mb_hi;
        sc = sc >> (8 * (is & 3));
        mbyte = mbyte >> (8 * (is & 3));
        sc &= 0x3F;
        mbyte &= 0x3F;

        const float d = loadd.x * float(sc);
        const float m = loadd.y * float(mbyte);
        shAscales[is * shAscales_stride + tid_row] = vec2(d,m);
    }

    barrier();
}
#endif

#endif

float16_t dequantFuncQ4_K(const in decodeBufQ4_K bl, const in uint blockCoords[2], const in uint coordInBlock[2])
{
    decodeBufQ4_K_packed16 bl16 = decodeBufQ4_K_packed16(bl);
    decodeBufQ4_K_packed128 bl128 = decodeBufQ4_K_packed128(bl);
    const uint idx = coordInBlock[1];

    const uint b = (idx & 0x20) >> 5;            // 0,1
    const uint is = (idx & 0xE0) >> 5;         // 0..7

#if defined(IS_MUL_MM2) && defined(DATA_A_Q4_K)
    vec2 v = shAscales[is * shAscales_stride + (blockCoords[0] % BM)];
    float d = v.x;
    float m = v.y;
#else
    uvec4 v = bl128.block.q4k[0];
    const vec2 loadd = vec2(unpackFloat2x16(v.x));

    uint32_t sc;
    uint32_t mbyte;

    uint32_t scale0 = v.y;
    uint32_t scale4 = v.z;
    uint32_t scale8 = v.w;

    uint32_t sc_lo = scale0;
    uint32_t mb_lo = scale4;
    uint32_t sc_hi = (scale8 & 0x0F0F0F0F) | ((scale0 & 0xC0C0C0C0) >> 2);
    uint32_t mb_hi = ((scale8 & 0xF0F0F0F0) >> 4) | ((scale4 & 0xC0C0C0C0) >> 2);

    sc = is < 4 ? sc_lo : sc_hi;
    mbyte = is < 4 ? mb_lo : mb_hi;
    sc = sc >> (8 * (is & 3));
    mbyte = mbyte >> (8 * (is & 3));
    sc &= 0x3F;
    mbyte &= 0x3F;

    const float d = loadd.x * float(sc);
    const float m = loadd.y * float(mbyte);
#endif

    uint qs = uint32_t(bl16.block.qs[((idx & 0xC0) >> 2) + ((idx & 0x1E) >> 1)]);
    qs = (qs >> (b * 4 + 8 * (idx & 1))) & 0xF;

    float ret = d * float(qs) - m;

    return float16_t(ret);
}

layout(buffer_reference, std430, buffer_reference_align = 16) buffer decodeBufQ5_K {
   block_q5_K block;
};

layout(buffer_reference, std430, buffer_reference_align = 16) buffer decodeBufQ5_K_packed16 {
   block_q5_K_packed16 block;
};

layout(buffer_reference, std430, buffer_reference_align = 16) buffer decodeBufQ5_K_packed128 {
   block_q5_K_packed128 block;
};

float16_t dequantFuncQ5_K(const in decodeBufQ5_K bl, const in uint blockCoords[2], const in uint coordInBlock[2])
{
    decodeBufQ5_K_packed16 bl16 = decodeBufQ5_K_packed16(bl);
    decodeBufQ5_K_packed128 bl128 = decodeBufQ5_K_packed128(bl);
    const uint idx = coordInBlock[1];

    const uint b = (idx & 0x20) >> 5;          // 0,1
    const uint is = (idx & 0xE0) >> 5;         // 0..7

#if defined(IS_MUL_MM2) && defined(DATA_A_Q5_K)
    vec2 v = shAscales[is * shAscales_stride + (blockCoords[0] % BM)];
    float d = v.x;
    float m = v.y;
#else
    uvec4 v = bl128.block.q5k[0];

    const f16vec2 loadd = unpackFloat2x16(v.x);

    uint32_t sc;
    uint32_t mbyte;

    uint32_t scale0 = v.y;
    uint32_t scale4 = v.z;
    uint32_t scale8 = v.w;

    uint32_t sc_lo = scale0;
    uint32_t mb_lo = scale4;
    uint32_t sc_hi = (scale8 & 0x0F0F0F0F) | ((scale0 & 0xC0C0C0C0) >> 2);
    uint32_t mb_hi = ((scale8 & 0xF0F0F0F0) >> 4) | ((scale4 & 0xC0C0C0C0) >> 2);

    sc = is < 4 ? sc_lo : sc_hi;
    mbyte = is < 4 ? mb_lo : mb_hi;
    sc = sc >> (8 * (is & 3));
    mbyte = mbyte >> (8 * (is & 3));
    sc &= 0x3F;
    mbyte &= 0x3F;

    const float16_t d = loadd.x * float16_t(sc);
    const float16_t m = loadd.y * float16_t(mbyte);
#endif

    uint qh = uint32_t(bl16.block.qh[(idx & 0x1E) >> 1]);
    qh = ((qh >> is) & 0x101) << 4;

    uint qs = uint32_t(bl16.block.qs[((idx & 0xC0) >> 2) + ((idx & 0x1E) >> 1)]);
    qs = (qs >> (b * 4)) & 0x0F0F;
    qs = unpack8(qs | qh)[idx & 1];

    float ret = d * float(qs) - m;

    return float16_t(ret);
}

layout(buffer_reference, std430, buffer_reference_align = 2) buffer decodeBufQ6_K {
   block_q6_K block;
};

layout(buffer_reference, std430, buffer_reference_align = 16) buffer decodeBufQ6_K_packed16 {
   block_q6_K_packed16 block;
};

float16_t dequantFuncQ6_K(const in decodeBufQ6_K bl, const in uint blockCoords[2], const in uint coordInBlock[2])
{
    decodeBufQ6_K_packed16 bl16 = decodeBufQ6_K_packed16(bl);
    const uint idx = coordInBlock[1];

    const uint b = (idx & 0x40) >> 6;           // 0,1
    const uint qhshift = (idx & 0x60) >> 4;    // 0,2,4,6
    const uint is = (idx & 0xF0) >> 4;          // 0..15

    const float16_t dscale = bl.block.d * float16_t(bl.block.scales[is]);

    uint ql = uint32_t(bl16.block.ql[((idx & 0x80) >> 2) + ((idx & 0x3E) >> 1)]);
    ql = (ql >> (b * 4)) & 0x0F0F;

    uint qh = uint32_t(bl16.block.qh[((idx & 0x80) >> 3) + ((idx & 0x1E) >> 1)]);
    qh = ((qh >> qhshift) & 0x0303) << 4;

    int q = unpack8(ql | qh)[idx & 1];

    float16_t ret = dscale * float16_t(q - 32);

    return ret;
}

#if defined(DATA_A_IQ1_S)
layout(buffer_reference, std430, buffer_reference_align = 2) buffer decodeBufIQ1_S {
   block_iq1_s block;
};

float16_t dequantFuncIQ1_S(const in decodeBufIQ1_S bl, const in uint blockCoords[2], const in uint coordInBlock[2])
{
    const float16_t d = bl.block.d;
    const uint idx = coordInBlock[1];

    const uint ib32 = (idx & 0xE0) >> 5;
    const uint ib8 = (idx & 0xF8) >> 3;

    const uint qh = bl.block.qh[ib32];
    const uint qs = bl.block.qs[ib8];
    const float dl = d * float(2 * bitfieldExtract(qh, 12, 3) + 1);
    const float delta = ((qh & 0x8000) != 0) ? -IQ1S_DELTA : IQ1S_DELTA;
    const uint grid = iq1s_grid[qs | (bitfieldExtract(qh, 3 * int(ib8 & 3), 3) << 8)];

    float16_t ret = float16_t(dl) * (float16_t(bitfieldExtract(int(grid), 2 * int(idx % 8), 2)) + float16_t(delta));
    return ret;
}
#endif

#if defined(DATA_A_IQ1_M)
layout(buffer_reference, std430, buffer_reference_align = 2) buffer decodeBufIQ1_M {
   block_iq1_m block;
};

layout(buffer_reference, std430, buffer_reference_align = 8) buffer decodeBufIQ1_M_packed64 {
   block_iq1_m_packed64 block;
};

float16_t dequantFuncIQ1_M(const in decodeBufIQ1_M bl, const in uint blockCoords[2], const in uint coordInBlock[2])
{
    decodeBufIQ1_M_packed64 bl64 = decodeBufIQ1_M_packed64(bl);
    const uint idx = coordInBlock[1];

    uvec2 scales = unpack32(bl64.block.scales);
    const float16_t d = uint16BitsToHalf(uint16_t(((scales.x & 0xF000) >> 12) | ((scales.x & 0xF0000000) >> 24) | ((scales.y & 0xF000) >> 4) | ((scales.y & 0xF0000000) >> 16)));

    const uint ib8 = (idx & 0xF8) >> 3;
    const uint ib16 = (idx & 0xF0) >> 4;
    const int i8 = int(idx % 8);
    const uint sc = bl.block.scales[ib8 / 8];
    const uint qs = bl.block.qs[ib8];
    const uint qh = bl.block.qh[ib16] >> (4 * (ib8 & 1));
    const float dl = 2 * bitfieldExtract(sc, 3 * int(ib16 & 3), 3) + 1;
    const float delta = ((qh & 8) != 0) ? -IQ1S_DELTA : IQ1S_DELTA;
    const uint grid = iq1s_grid[qs | ((qh & 7) << 8)];

    float16_t ret = d * float16_t(dl) * (float16_t(bitfieldExtract(int(grid), 2 * i8, 2)) + float16_t(delta));
    return ret;
}
#endif

#if defined(DATA_A_IQ2_XXS)
layout(buffer_reference, std430, buffer_reference_align = 2) buffer decodeBufIQ2_XXS {
   block_iq2_xxs block;
};

layout(buffer_reference, std430, buffer_reference_align = 2) buffer decodeBufIQ2_XXS_packed16 {
   block_iq2_xxs_packed16 block;
};

float16_t dequantFuncIQ2_XXS(const in decodeBufIQ2_XXS bl, const in uint blockCoords[2], const in uint coordInBlock[2])
{
    decodeBufIQ2_XXS_packed16 bl16 = decodeBufIQ2_XXS_packed16(bl);
    const float16_t d = bl.block.d;
    const uint idx = coordInBlock[1];

    const uint ib32 = (idx & 0xE0) >> 5; // 0..7
    const uint ib8 = (idx & 0x18) >> 3;  // 0..3
    const uint iqs = 8 * ib32 + ib8;

    const uint qs = bl.block.qs[iqs];
    const uint signscale = pack32(u16vec2(bl16.block.qs[4*ib32+2], bl16.block.qs[4*ib32+3]));

    const float dscale = float(bl.block.d) * 0.25 * (0.5 + float(signscale >> 28));
    uint sign = bitfieldExtract(signscale, 7 * int(ib8), 7);
    sign |= bitCount(sign) << 7;

    uint g2 = iq2xxs_grid[qs][(idx & 4) >> 2];
    g2 >>= (idx & 2) * 8;
    const vec2 g = vec2(unpack8(g2));

    vec2 ret = dscale * g * ((sign & (1 << (idx & 7))) != 0 ? -1.0hf : 1.0hf);
    return float16_t(ret[idx & 1]);
}
#endif

#if defined(DATA_A_IQ2_XS)
layout(buffer_reference, std430, buffer_reference_align = 2) buffer decodeBufIQ2_XS {
   block_iq2_xs block;
};

float16_t dequantFuncIQ2_XS(const in decodeBufIQ2_XS bl, const in uint blockCoords[2], const in uint coordInBlock[2])
{
    const float16_t d = bl.block.d;
    const uint idx = coordInBlock[1];

    const uint is = (idx & 0xE0) >> 5;     // 0..8
    const uint sshift = (idx & 0x10) >> 2; // 0,4
    const uint iqs = (idx & 0xF8) >> 3;    // 0..63

    const uint16_t qs = bl.block.qs[iqs];
    const float dscale = float(bl.block.d) * 0.25 * (0.5 + float((bl.block.scales[is] >> sshift) & 0xF));

    uint sign = uint(qs >> 9);
    sign |= bitCount(sign) << 7;
    uint g2 = iq2xs_grid[qs & 0x1FF][(idx & 4) >> 2];
    g2 >>= (idx & 2) * 8;
    const vec2 g = vec2(unpack8(g2));

    vec2 ret = dscale * g * ((sign & (1 << (idx & 7))) != 0 ? -1.0hf : 1.0hf);
    return float16_t(ret[idx & 1]);
}
#endif

#if defined(DATA_A_IQ2_S)
layout(buffer_reference, std430, buffer_reference_align = 2) buffer decodeBufIQ2_S {
   block_iq2_s block;
};

float16_t dequantFuncIQ2_S(const in decodeBufIQ2_S bl, const in uint blockCoords[2], const in uint coordInBlock[2])
{
    uint idx = coordInBlock[1];

    const uint ib32 = (idx & 0xE0) >> 5;        // 0..7
    const uint ib8 = (idx & 0xF8) >> 3;         // 0..31
    const uint qhshift = 2 * (ib8 % 4);

    const uint scale = (bl.block.scales[ib32] >> ((idx & 0x10) >> 2)) & 0xf;
    const uint qs = bl.block.qs[ib8];
    const uint qh = bl.block.qh[ib32];
    const uint sign = bl.block.qs[QUANT_K / 8 + ib8] >> (idx & 0x6);

    const float d = float(bl.block.d);
    const float db = d * 0.25 * (0.5 + scale);
    const ivec2 sign01 = 1 - (2 & ivec2(sign << 1, sign));
    uint g2 = iq2s_grid[qs | ((qh << (8 - qhshift)) & 0x300)][(idx & 4) >> 2];
    g2 >>= (idx & 2) * 8;
    const vec2 v = db * vec2(sign01) * vec2(unpack8(g2));
    return float16_t(v[idx & 1]);
}
#endif

#if defined(DATA_A_IQ3_XXS)
layout(buffer_reference, std430, buffer_reference_align = 2) buffer decodeBufIQ3_XXS {
   block_iq3_xxs block;
};

layout(buffer_reference, std430, buffer_reference_align = 2) buffer decodeBufIQ3_XXS_packed16 {
   block_iq3_xxs_packed16 block;
};

float16_t dequantFuncIQ3_XXS(const in decodeBufIQ3_XXS bl, const in uint blockCoords[2], const in uint coordInBlock[2])
{
    decodeBufIQ3_XXS_packed16 bl16 = decodeBufIQ3_XXS_packed16(bl);
    uint idx = coordInBlock[1];

    const uint iqs = (idx & 0xFC) >> 2;             // 0..63
    const uint is = QUANT_K / 4 + ((idx & 0xE0) >> 3);// 8 values

    const float d = float(bl.block.d);
    const uint qs = bl.block.qs[iqs];
    const uint signs = pack32(u16vec2(
        bl16.block.qs[is/2+0],
        bl16.block.qs[is/2+1]
    ));
    const float db = d * 0.5 * (0.5 + (signs >> 28));
    const uint32_t sign7 = bitfieldExtract(signs, 7 * (int(iqs / 2) % 4), 7);
    const uint sign = (sign7 | (bitCount(sign7) << 7)) >> (idx & 0x6);
    const ivec2 sign01 = ivec2(1 - (2 & ivec2(sign << 1, sign)));
    const uint grid = iq3xxs_grid[qs] >> (16 * ((idx & 2) >> 1));
    const vec2 v = db * vec2(sign01) * vec2(unpack8(grid).xy);
    return float16_t(v[idx & 1]);
}
#endif

#if defined(DATA_A_IQ3_S)
layout(buffer_reference, std430, buffer_reference_align = 2) buffer decodeBufIQ3_S {
   block_iq3_s block;
};

float16_t dequantFuncIQ3_S(const in decodeBufIQ3_S bl, const in uint blockCoords[2], const in uint coordInBlock[2])
{
    uint idx = coordInBlock[1];

    const uint iqs = (idx & 0xFC) >> 2;           // 0..63
    const uint iqh = (idx & 0xE0) >> 5;

    const float d = float(bl.block.d);
    const uint qs = bl.block.qs[iqs];
    const uint qh = bl.block.qh[iqh];
    const int8_t sign = int8_t(bl.block.signs[iqs / 2] >> (idx & 0x6));
    const uint scale = bl.block.scales[iqs / 16];
    const ivec2 sign01 = ivec2(1 - (2 & ivec2(sign << 1, sign)));
    const float db = d * (1 + 2 * ((scale >> (4 * (iqh & 1))) & 0xf));
    const uint32_t grid = iq3s_grid[qs | ((qh << (8 - (iqs % 8))) & 256)] >> ((idx & 2) << 3);
    const vec2 v = db * vec2(sign01) * vec2(unpack8(grid).xy);

    return float16_t(v[idx & 1]);
}
#endif

#if defined(DATA_A_IQ4_XS)
layout(buffer_reference, std430, buffer_reference_align = 2) buffer decodeBufIQ4_XS {
   block_iq4_xs block;
};

float16_t dequantFuncIQ4_XS(const in decodeBufIQ4_XS bl, const in uint blockCoords[2], const in uint coordInBlock[2])
{
    const float16_t d = bl.block.d;
    const uint idx = coordInBlock[1];

    const uint ib32 = (idx & 0xE0) >> 5; // 0..7

    const uint sl = (bl.block.scales_l[ib32/2] >> (4 * (ib32 & 1))) & 0xF;
    const uint sh = ((bl.block.scales_h) >> (2 * ib32)) & 3;
    const uint qshift = (idx & 16) >> 2;
    const uint q = (bl.block.qs[16 * ib32 + (idx % 16)] >> qshift) & 0xF;

    float16_t ret = d * float16_t(int(sl | (sh << 4)) - 32) * float16_t(kvalues_iq4nl[q]);
    return ret;
}
#endif

#if defined(DATA_A_IQ4_NL)
layout(buffer_reference, std430, buffer_reference_align = 2) buffer decodeBufIQ4_NL {
   block_iq4_nl block;
};

float16_t dequantFuncIQ4_NL(const in decodeBufIQ4_NL bl, const in uint blockCoords[2], const in uint coordInBlock[2])
{
    const float16_t d = bl.block.d;
    const uint idx = coordInBlock[1];
    const uint iqs = idx & 0xF;
    const uint shift = (idx & 0x10) >> 2;
    uint32_t qs = bl.block.qs[iqs];
    qs >>= shift;
    qs &= 0xF;
    float16_t ret = float16_t(kvalues_iq4nl[qs]) * d;
    return ret;
}
#endif

#if defined(DATA_A_MXFP4)
layout(buffer_reference, std430, buffer_reference_align = 2) buffer decodeBufMXFP4 {
   block_mxfp4 block;
};

float16_t dequantFuncMXFP4(const in decodeBufMXFP4 bl, const in uint blockCoords[2], const in uint coordInBlock[2])
{
    const float d = e8m0_to_fp32(bl.block.e);
    const uint idx = coordInBlock[1];
    const uint iqs = idx & 0xF;
    const uint shift = (idx & 0x10) >> 2;
    uint32_t qs = bl.block.qs[iqs];
    qs >>= shift;
    qs &= 0xF;
    float16_t ret = float16_t(kvalues_mxfp4[qs] * d * 0.5);
    return ret;
}
#endif

// ── Planar / Iso quant (planar3, iso3, planar4, iso4): same constants as CUDA planar-iso-constants.cuh / flash_attn_base.glsl
#if defined(DATA_A_PLANAR3) || defined(DATA_A_ISO3) || defined(DATA_A_PLANAR4) || defined(DATA_A_ISO4)

const float PI_CENTROIDS_3BIT[8] = float[8](
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
);
const float PI_CENTROIDS_4BIT[16] = float[16](
    -0.173926f, -0.117195f, -0.089527f, -0.068756f,
    -0.051262f, -0.035597f, -0.020989f, -0.006938f,
     0.006938f,  0.020989f,  0.035597f,  0.051262f,
     0.068756f,  0.089527f,  0.117195f,  0.173926f
);
const float PI_COS[64] = float[64](
    -0.9095053397f, 0.1535578452f, -0.8537489227f, -0.6827218011f,
    -0.4249387949f, 0.9864510046f, 0.9906673944f, 0.5752363372f,
    -0.9866459035f, 0.9878848090f, -0.6215683804f, -0.9835597698f,
     0.8777263755f, -0.4624640047f, 0.2843135922f, -0.7739960698f,
     0.2385234222f, 0.9121914932f, -0.8815003943f, -0.2639699512f,
    -0.5517087300f, -0.9035294557f, -0.8520543188f, -0.5600635985f,
    -0.7667286376f, -0.9877949369f, -0.9781949787f, -0.9953372831f,
    -0.8622053901f, -0.7382118186f, 0.9136037642f, -0.2558504503f,
    -0.8541000475f, -0.6159335408f, 0.9861256679f, -0.6758560284f,
     0.4249571682f, -0.6219544719f, 0.9130573430f, -0.5948161096f,
     0.5759782996f, 0.9729901203f, 0.6535998325f, 0.9222195491f,
    -0.7668084044f, 0.5116178563f, -0.7848786574f, 0.9902111051f,
     0.1997167840f, 0.7173003220f, -0.9999998006f, -0.9557868691f,
     0.5594852693f, -0.9980111824f, 0.9782398557f, -0.9150004329f,
    -0.4084754305f, 0.0071549185f, 0.9558482753f, -0.0971921648f,
    -0.9469334002f, 0.9999492419f, 0.6100589016f, 0.0350818915f
);
const float PI_SIN[64] = float[64](
    -0.4156922383f, 0.9881396603f, 0.5206849114f, -0.7306784124f,
    -0.9052220836f, 0.1640561354f, 0.1363015542f, 0.8179872593f,
     0.1628798979f, 0.1551889303f, 0.7833599099f, -0.1805828875f,
    -0.4791621957f, 0.8866380571f, -0.9587313395f, 0.6331904010f,
    -0.9711367448f, 0.4097641756f, 0.4721832852f, -0.9645309040f,
     0.8340368561f, 0.4285259884f, 0.5234533769f, 0.8284496156f,
     0.6419713361f, -0.1557599517f, -0.2076886701f, 0.0964556523f,
     0.5065588468f, -0.6745689815f, -0.4066056591f, -0.9667163736f,
     0.5201087471f, -0.7877981171f, 0.1660005034f, -0.7370336688f,
     0.9052134584f, 0.7830534049f, -0.4078312009f, -0.8038618014f,
     0.8174649829f, -0.2308467584f, -0.7568403127f, -0.3866666566f,
     0.6418760557f, -0.8592131104f, 0.6196494922f, 0.1395778183f,
     0.9798536657f, 0.6967641265f, -0.0006314605f, 0.2940603015f,
     0.8288402943f, -0.0630371303f, 0.2074771907f, 0.4034528570f,
     0.9127693152f, -0.9999744032f, 0.2938606379f, 0.9952656344f,
     0.3214298299f, 0.0100754012f, -0.7923560668f, -0.9993844410f
);
const float PI_QW[32] = float[32](
     0.8350809813f, -0.1648498178f, 0.1283752173f, 0.2897698581f,
    -0.1820549369f, 0.9549587369f, -0.8741137385f, 0.8988990188f,
    -0.1312584430f, -0.3990598321f, -0.2694816887f, -0.1181898862f,
     0.1363395452f, 0.2665117681f, -0.8263269663f, -0.1834189594f,
     0.3098247349f, 0.2804697454f, -0.5655074716f, -0.1627507508f,
     0.8684155941f, 0.2233296037f, -0.1291671842f, 0.6606932878f,
    -0.5694432259f, -0.2782760859f, 0.5113853812f, -0.5139024258f,
     0.7489815354f, -0.3037399948f, -0.4143463373f, -0.3524050117f
);
const float PI_QX[32] = float[32](
     0.3547102809f, -0.5782636404f, -0.8299785256f, 0.5694668293f,
    -0.8199930191f, 0.1259543896f, -0.3090814352f, -0.2613596618f,
    -0.1660282463f, -0.5143862963f, 0.5898610353f, -0.8277072310f,
    -0.6826571226f, -0.1740629375f, 0.1416199356f, 0.4648889899f,
     0.3485621810f, 0.8982698917f, -0.3015249372f, 0.4990116358f,
     0.2398942262f, -0.7447698116f, 0.4783197045f, 0.0735855624f,
    -0.2975912094f, -0.0700704753f, 0.2975627482f, -0.2652103305f,
    -0.1539765000f, 0.0849994123f, -0.1069803685f, -0.5753474832f
);
const float PI_QY[32] = float[32](
     0.2416850179f, -0.4488199651f, 0.3478420675f, 0.5024775267f,
     0.1696543097f, 0.1760476083f, 0.0254505407f, 0.2389279008f,
    -0.9429193735f, 0.3925755024f, -0.2757458389f, -0.1485267133f,
     0.5530825853f, -0.8936085105f, 0.2953715622f, -0.5285226703f,
     0.7939327955f, 0.0139789311f, -0.2555710375f, 0.4543992281f,
    -0.2698826790f, -0.4736968279f, 0.4361720681f, -0.3461222053f,
     0.0792116225f, 0.8827795386f, 0.7416539788f, -0.3826399446f,
    -0.3534849286f, -0.8696597815f, -0.6908422709f, 0.2082736641f
);
const float PI_QZ[32] = float[32](
     0.3038694561f, 0.4734756052f, -0.3878843784f, 0.5831694603f,
    -0.5054479241f, -0.1731694490f, -0.3737666607f, 0.2328704894f,
     0.2621760964f, 0.6239953637f, -0.7082104683f, 0.5308507681f,
    -0.4413037896f, -0.2802782655f, -0.4522367120f, -0.6698107123f,
    -0.3752456903f, -0.3359423280f, 0.7181019187f, 0.7106907368f,
     0.3100073636f, 0.4016827941f, 0.7350437641f, -0.6607965231f,
     0.7619289756f, 0.3648703992f, -0.3040413559f, 0.7213236690f,
     0.5280022621f, -0.3742936850f, -0.5760775208f, 0.7015634775f
);

uint unpack_planar_iso3_idx(uint8_t qs, uint8_t signs, uint j) {
    uint low = (uint(qs) >> ((j % 4) * 2)) & 0x3;
    uint hi = (uint(signs) >> (j % 8)) & 0x1;
    return low | (hi << 2);
}

#endif // shared planar/iso constants

#if defined(DATA_A_PLANAR3)
layout(buffer_reference, std430, buffer_reference_align = 4) buffer decodeBufPlanar3 {
   block_planar3_0 block;
};

float16_t dequantFuncPlanar3(const in decodeBufPlanar3 bl, const in uint blockCoords[2], const in uint coordInBlock[2])
{
    const float norm = float(bl.block.norm);
    const uint idx = coordInBlock[1];
    const uint q0i = idx & ~1u;
    const uint q1i = q0i + 1;

    const uint8_t qs0 = bl.block.qs[q0i / 4];
    const uint8_t sg0 = bl.block.signs[q0i / 8];
    const uint8_t qs1 = bl.block.qs[q1i / 4];
    const uint8_t sg1 = bl.block.signs[q1i / 8];

    float q0 = PI_CENTROIDS_3BIT[unpack_planar_iso3_idx(qs0, sg0, q0i)];
    float q1 = PI_CENTROIDS_3BIT[unpack_planar_iso3_idx(qs1, sg1, q1i)];

    const uint p = q0i / 2;
    const float c = PI_COS[p];
    const float s = PI_SIN[p];
    const float r0 = (c * q0 + s * q1) * norm;
    const float r1 = (-s * q0 + c * q1) * norm;
    return float16_t((idx & 1u) != 0u ? r1 : r0);
}
#endif

#if defined(DATA_A_ISO3)
layout(buffer_reference, std430, buffer_reference_align = 4) buffer decodeBufIso3 {
   block_iso3_0 block;
};

float16_t dequantFuncIso3(const in decodeBufIso3 bl, const in uint blockCoords[2], const in uint coordInBlock[2])
{
    const float norm = float(bl.block.norm);
    const uint idx = coordInBlock[1];
    const uint g = idx / 4;

    float qvals[4];
    for (int c = 0; c < 4; ++c) {
        const uint j = g * 4u + uint(c);
        const uint8_t qs_byte = bl.block.qs[j / 4];
        const uint8_t signs_byte = bl.block.signs[j / 8];
        qvals[c] = PI_CENTROIDS_3BIT[unpack_planar_iso3_idx(qs_byte, signs_byte, j)];
    }

    const float qw = PI_QW[g], qx = -PI_QX[g], qy = -PI_QY[g], qz = -PI_QZ[g];
    const float rw = qw*qvals[0] - qx*qvals[1] - qy*qvals[2] - qz*qvals[3];
    const float rx = qw*qvals[1] + qx*qvals[0] + qy*qvals[3] - qz*qvals[2];
    const float ry = qw*qvals[2] - qx*qvals[3] + qy*qvals[0] + qz*qvals[1];
    const float rz = qw*qvals[3] + qx*qvals[2] - qy*qvals[1] + qz*qvals[0];

    const uint offset = idx % 4;
    const float results[4] = float[4](rw, rx, ry, rz);
    return float16_t(results[offset] * norm);
}
#endif

#if defined(DATA_A_PLANAR4)
layout(buffer_reference, std430, buffer_reference_align = 4) buffer decodeBufPlanar4 {
   block_planar4_0 block;
};

float16_t dequantFuncPlanar4(const in decodeBufPlanar4 bl, const in uint blockCoords[2], const in uint coordInBlock[2])
{
    const float norm = float(bl.block.norm);
    const uint idx = coordInBlock[1];
    const uint q0i = idx & ~1u;
    const uint q1i = q0i + 1;

    const uint idx_nib0 = (bl.block.qs[q0i / 2] >> ((q0i & 1u) * 4)) & 0xFu;
    const uint idx_nib1 = (bl.block.qs[q1i / 2] >> ((q1i & 1u) * 4)) & 0xFu;
    float q0 = PI_CENTROIDS_4BIT[idx_nib0];
    float q1 = PI_CENTROIDS_4BIT[idx_nib1];

    const uint p = q0i / 2;
    const float c = PI_COS[p];
    const float s = PI_SIN[p];
    const float r0 = (c * q0 + s * q1) * norm;
    const float r1 = (-s * q0 + c * q1) * norm;
    return float16_t((idx & 1u) != 0u ? r1 : r0);
}
#endif

#if defined(DATA_A_ISO4)
layout(buffer_reference, std430, buffer_reference_align = 4) buffer decodeBufIso4 {
   block_iso4_0 block;
};

float16_t dequantFuncIso4(const in decodeBufIso4 bl, const in uint blockCoords[2], const in uint coordInBlock[2])
{
    const float norm = float(bl.block.norm);
    const uint idx = coordInBlock[1];
    const uint g = idx / 4;

    float qvals[4];
    for (int c = 0; c < 4; ++c) {
        const uint j = g * 4u + uint(c);
        const uint idx_nib = (bl.block.qs[j / 2] >> ((j & 1u) * 4)) & 0xFu;
        qvals[c] = PI_CENTROIDS_4BIT[idx_nib];
    }

    const float qw = PI_QW[g], qx = -PI_QX[g], qy = -PI_QY[g], qz = -PI_QZ[g];
    const float rw = qw*qvals[0] - qx*qvals[1] - qy*qvals[2] - qz*qvals[3];
    const float rx = qw*qvals[1] + qx*qvals[0] + qy*qvals[3] - qz*qvals[2];
    const float ry = qw*qvals[2] - qx*qvals[3] + qy*qvals[0] + qz*qvals[1];
    const float rz = qw*qvals[3] + qx*qvals[2] - qy*qvals[1] + qz*qvals[0];

    const uint offset = idx % 4;
    const float results[4] = float[4](rw, rx, ry, rz);
    return float16_t(results[offset] * norm);
}
#endif

#if defined(DATA_A_Q4_0)
#define dequantFuncA dequantFuncQ4_0
#elif defined(DATA_A_Q4_1)
#define dequantFuncA dequantFuncQ4_1
#elif defined(DATA_A_Q5_0)
#define dequantFuncA dequantFuncQ5_0
#elif defined(DATA_A_Q5_1)
#define dequantFuncA dequantFuncQ5_1
#elif defined(DATA_A_Q8_0)
#define dequantFuncA dequantFuncQ8_0
#elif defined(DATA_A_Q2_K)
#define dequantFuncA dequantFuncQ2_K
#elif defined(DATA_A_Q3_K)
#define dequantFuncA dequantFuncQ3_K
#elif defined(DATA_A_Q4_K)
#define dequantFuncA dequantFuncQ4_K
#define fetch_scales fetch_scalesQ4_K
#define store_scales store_scalesQ4_K
#elif defined(DATA_A_Q5_K)
#define dequantFuncA dequantFuncQ5_K
#define fetch_scales fetch_scalesQ5_K
#define store_scales store_scalesQ4_K
#elif defined(DATA_A_Q6_K)
#define dequantFuncA dequantFuncQ6_K
#elif defined(DATA_A_IQ1_S)
#define dequantFuncA dequantFuncIQ1_S
#elif defined(DATA_A_IQ1_M)
#define dequantFuncA dequantFuncIQ1_M
#elif defined(DATA_A_IQ2_XXS)
#define dequantFuncA dequantFuncIQ2_XXS
#elif defined(DATA_A_IQ2_XS)
#define dequantFuncA dequantFuncIQ2_XS
#elif defined(DATA_A_IQ2_S)
#define dequantFuncA dequantFuncIQ2_S
#elif defined(DATA_A_IQ3_XXS)
#define dequantFuncA dequantFuncIQ3_XXS
#elif defined(DATA_A_IQ3_S)
#define dequantFuncA dequantFuncIQ3_S
#elif defined(DATA_A_IQ4_XS)
#define dequantFuncA dequantFuncIQ4_XS
#elif defined(DATA_A_IQ4_NL)
#define dequantFuncA dequantFuncIQ4_NL
#elif defined(DATA_A_MXFP4)
#define dequantFuncA dequantFuncMXFP4
#elif defined(DATA_A_PLANAR3)
#define dequantFuncA dequantFuncPlanar3
#elif defined(DATA_A_ISO3)
#define dequantFuncA dequantFuncIso3
#elif defined(DATA_A_PLANAR4)
#define dequantFuncA dequantFuncPlanar4
#elif defined(DATA_A_ISO4)
#define dequantFuncA dequantFuncIso4
#elif defined(DATA_A_F32)
#define dequantFuncA dequantFuncF32
#endif
