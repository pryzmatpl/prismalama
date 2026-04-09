
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout (constant_id =  0) const uint32_t WorkGroupSize = 128;
layout (constant_id =  1) const uint32_t Br = 1;
layout (constant_id =  2) const uint32_t Bc = 32;
layout (constant_id =  3) const uint32_t HSK = 32;
layout (constant_id =  4) const uint32_t HSV = 32;
layout (constant_id =  5) const uint32_t Clamp = 0;
layout (constant_id =  6) const uint32_t D_split = 16;
layout (constant_id =  7) const uint32_t row_split = 1;
layout (constant_id =  8) const uint32_t SubGroupSize = 32;
layout (constant_id =  9) const uint32_t SHMEM_STAGING = 0;
layout (constant_id = 10) const uint32_t Flags = 0;
layout (constant_id = 11) const uint32_t LIMIT_OCCUPANCY_SHMEM = 0;

const bool USE_MASK_OPT    = (Flags & 1) != 0;
const bool MASK_ENABLE     = (Flags & 2) != 0;
const bool LOGIT_SOFTCAP   = (Flags & 4) != 0;
const bool OLD_AMD_WINDOWS = (Flags & 8) != 0;

// Round up head sizes to a multiple of 16, for coopmat1/coopmat2 paths
const uint32_t HSK_pad = (HSK + 15) & ~15;
const uint32_t HSV_pad = (HSV + 15) & ~15;

const bool KV_bounds_check = Clamp != 0;

layout (push_constant) uniform parameter {
    uint32_t N;
    uint32_t KV;

    uint32_t ne1;
    uint32_t ne2;
    uint32_t ne3;

    uint32_t neq2;
    uint32_t neq3;
    uint32_t nek2;
    uint32_t nek3;
    uint32_t nev2;
    uint32_t nev3;
    uint32_t nem1;
    uint32_t nem2;
    uint32_t nem3;

    uint32_t nb01;
    uint32_t nb02;
    uint32_t nb03;
    uint32_t nb11;
    uint32_t nb12;
    uint32_t nb13;
    uint32_t nb21;
    uint32_t nb22;
    uint32_t nb23;

    float scale;
    float max_bias;
    float logit_softcap;

    uint32_t mask_n_head_log2;
    float m0;
    float m1;

    uint32_t gqa_ratio;
    uint32_t split_kv;
    uint32_t k_num;
} p;

#define SINK_ENABLE_BIT (1<<24)
#define N_LOG2_MASK 0xFFFF

layout (binding = 4) readonly buffer S {float data_s[];};

layout (binding = 5) writeonly buffer O {D_TYPE data_o[];};
layout (binding = 5) writeonly buffer OV4 {D_TYPEV4 data_ov4[];};

layout (binding = 6) readonly buffer MO {uint32_t data_mask_opt[];};

#define MASK_OPT_ALL_NEG_INF 1
#define MASK_OPT_ALL_ZERO 2

#define BINDING_IDX_K 0
#define BINDING_IDX_V 1
#if defined(DATA_A_F32)
layout (binding = 1) readonly buffer K_PACKED {vec4 k_data_packed[];} k_packed;
layout (binding = 2) readonly buffer V_PACKED {vec4 v_data_packed[];} v_packed;
#elif defined(A_TYPE_PACKED16)
layout (binding = 1) readonly buffer K_PACKED16 {A_TYPE_PACKED16 k_data_packed16[];} k_packed;
layout (binding = 2) readonly buffer V_PACKED16 {A_TYPE_PACKED16 v_data_packed16[];} v_packed;
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1
#endif

#if defined(DATA_A_F32)
#undef BLOCK_SIZE
#define BLOCK_SIZE 4
#define BLOCK_BYTE_SIZE 16

FLOAT_TYPEV4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    // iqs is currently always zero in the flash attention shaders
    if (binding_idx == BINDING_IDX_K) {
        return FLOAT_TYPEV4(k_packed.k_data_packed[a_offset + ib]);
    } else {
        return FLOAT_TYPEV4(v_packed.v_data_packed[a_offset + ib]);
    }
}
#endif

#if defined(DATA_A_Q4_0)
#define BLOCK_BYTE_SIZE 18

FLOAT_TYPEV4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    if (binding_idx == BINDING_IDX_K) {
        uint vui_lo = uint(k_packed.k_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);
        uint vui_hi = uint(k_packed.k_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);
        uint shift = (iqs & 0x10) >> 2;
        vui_lo >>= shift;
        vui_hi >>= shift;

        return FLOAT_TYPE(k_packed.k_data_packed16[a_offset + ib].d) * (FLOAT_TYPEV4(vui_lo & 0xF, (vui_lo >> 8) & 0xF, vui_hi & 0xF, (vui_hi >> 8) & 0xF) - FLOAT_TYPE(8.0f));
    } else {
        uint vui_lo = uint(v_packed.v_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);
        uint vui_hi = uint(v_packed.v_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);
        uint shift = (iqs & 0x10) >> 2;
        vui_lo >>= shift;
        vui_hi >>= shift;

        return FLOAT_TYPE(v_packed.v_data_packed16[a_offset + ib].d) * (FLOAT_TYPEV4(vui_lo & 0xF, (vui_lo >> 8) & 0xF, vui_hi & 0xF, (vui_hi >> 8) & 0xF) - FLOAT_TYPE(8.0f));
    }
}
#endif

#if defined(DATA_A_Q8_0)
#define BLOCK_BYTE_SIZE 34
FLOAT_TYPEV4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    if (binding_idx == BINDING_IDX_K) {
        const i8vec2 v0 = unpack8(int32_t(k_packed.k_data_packed16[a_offset + ib].qs[iqs / 2])).xy; // vec4 used due to #12147
        const i8vec2 v1 = unpack8(int32_t(k_packed.k_data_packed16[a_offset + ib].qs[iqs / 2 + 1])).xy;

        return FLOAT_TYPE(k_packed.k_data_packed16[a_offset + ib].d) * FLOAT_TYPEV4(v0.x, v0.y, v1.x, v1.y);
    } else {
        const i8vec2 v0 = unpack8(int32_t(v_packed.v_data_packed16[a_offset + ib].qs[iqs / 2])).xy; // vec4 used due to #12147
        const i8vec2 v1 = unpack8(int32_t(v_packed.v_data_packed16[a_offset + ib].qs[iqs / 2 + 1])).xy;

        return FLOAT_TYPE(v_packed.v_data_packed16[a_offset + ib].d) * FLOAT_TYPEV4(v0.x, v0.y, v1.x, v1.y);
    }
}
#endif

// PlanarQuant 3-bit: 2D Givens rotation + 2-bit quantized + 1-bit QJL
// 3-bit centroids (same as turbo3)
const float PLANAR_CENTROIDS_3BIT[8] = float[8](
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
);

// Givens rotation: cos/sin for 64 pairs (seed=42, LCG PRNG)
const float PLANAR_COS[64] = float[64](
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
const float PLANAR_SIN[64] = float[64](
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

#if defined(DATA_A_PLANAR3)
#define BLOCK_BYTE_SIZE 50

uint unpack_planar3_idx(uint8_t qs, uint8_t signs, uint j) {
    uint low = (uint(qs) >> ((j % 4) * 2)) & 0x3;
    uint hi = (uint(signs) >> (j % 8)) & 0x1;
    return low | (hi << 2);
}

FLOAT_TYPEV4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    uint j0 = iqs;

    uint8_t qs = binding_idx == BINDING_IDX_K
        ? k_packed.k_data_packed16[a_offset + ib].qs[j0 / 4]
        : v_packed.v_data_packed16[a_offset + ib].qs[j0 / 4];
    uint8_t signs = binding_idx == BINDING_IDX_K
        ? k_packed.k_data_packed16[a_offset + ib].signs[j0 / 8]
        : v_packed.v_data_packed16[a_offset + ib].signs[j0 / 8];
    float norm = float(binding_idx == BINDING_IDX_K
        ? k_packed.k_data_packed16[a_offset + ib].norm
        : v_packed.v_data_packed16[a_offset + ib].norm);

    uint idx0 = unpack_planar3_idx(qs, signs, j0);
    uint idx1 = unpack_planar3_idx(qs, signs, j0 + 1);
    uint idx2 = unpack_planar3_idx(qs, signs, j0 + 2);
    uint idx3 = unpack_planar3_idx(qs, signs, j0 + 3);

    float q0 = PLANAR_CENTROIDS_3BIT[idx0];
    float q1 = PLANAR_CENTROIDS_3BIT[idx1];
    float q2 = PLANAR_CENTROIDS_3BIT[idx2];
    float q3 = PLANAR_CENTROIDS_3BIT[idx3];

    int p0 = j0 / 2;
    float c0 = PLANAR_COS[p0], s0 = PLANAR_SIN[p0];
    float r0 = (c0 * q0 + s0 * q1) * norm;
    float r1 = (-s0 * q0 + c0 * q1) * norm;

    int p1 = (j0 + 2) / 2;
    float c1 = PLANAR_COS[p1], s1 = PLANAR_SIN[p1];
    float r2 = (c1 * q2 + s1 * q3) * norm;
    float r3 = (-s1 * q2 + c1 * q3) * norm;

    return FLOAT_TYPEV4(r0, r1, r2, r3);
}
#endif

// IsoQuant 3-bit: quaternion 4D rotation + 2-bit quantized + 1-bit QJL
const float ISO_CENTROIDS_3BIT[8] = float[8](
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
);

const float ISO_QW[32] = float[32](
     0.8350809813f, -0.1648498178f, 0.1283752173f, 0.2897698581f,
    -0.1820549369f, 0.9549587369f, -0.8741137385f, 0.8988990188f,
    -0.1312584430f, -0.3990598321f, -0.2694816887f, -0.1181898862f,
     0.1363395452f, 0.2665117681f, -0.8263269663f, -0.1834189594f,
     0.3098247349f, 0.2804697454f, -0.5655074716f, -0.1627507508f,
     0.8684155941f, 0.2233296037f, -0.1291671842f, 0.6606932878f,
    -0.5694432259f, -0.2782760859f, 0.5113853812f, -0.5139024258f,
     0.7489815354f, -0.3037399948f, -0.4143463373f, -0.3524050117f
);
const float ISO_QX[32] = float[32](
     0.3547102809f, -0.5782636404f, -0.8299785256f, 0.5694668293f,
    -0.8199930191f, 0.1259543896f, -0.3090814352f, -0.2613596618f,
    -0.1660282463f, -0.5143862963f, 0.5898610353f, -0.8277072310f,
    -0.6826571226f, -0.1740629375f, 0.1416199356f, 0.4648889899f,
     0.3485621810f, 0.8982698917f, -0.3015249372f, 0.4990116358f,
     0.2398942262f, -0.7447698116f, 0.4783197045f, 0.0735855624f,
    -0.2975912094f, -0.0700704753f, 0.2975627482f, -0.2652103305f,
    -0.1539765000f, 0.0849994123f, -0.1069803685f, -0.5753474832f
);
const float ISO_QY[32] = float[32](
     0.2416850179f, -0.4488199651f, 0.3478420675f, 0.5024775267f,
     0.1696543097f, 0.1760476083f, 0.0254505407f, 0.2389279008f,
    -0.9429193735f, 0.3925755024f, -0.2757458389f, -0.1485267133f,
     0.5530825853f, -0.8936085105f, 0.2953715622f, -0.5285226703f,
     0.7939327955f, 0.0139789311f, -0.2555710375f, 0.4543992281f,
    -0.2698826790f, -0.4736968279f, 0.4361720681f, -0.3461222053f,
     0.0792116225f, 0.8827795386f, 0.7416539788f, -0.3826399446f,
    -0.3534849286f, -0.8696597815f, -0.6908422709f, 0.2082736641f
);
const float ISO_QZ[32] = float[32](
     0.3038694561f, 0.4734756052f, -0.3878843784f, 0.5831694603f,
    -0.5054479241f, -0.1731694490f, -0.3737666607f, 0.2328704894f,
     0.2621760964f, 0.6239953637f, -0.7082104683f, 0.5308507681f,
    -0.4413037896f, -0.2802782655f, -0.4522367120f, -0.6698107123f,
    -0.3752456903f, -0.3359423280f, 0.7181019187f, 0.7106907368f,
     0.3100073636f, 0.4016827941f, 0.7350437641f, -0.6607965231f,
     0.7619289756f, 0.3648703992f, -0.3040413559f, 0.7213236690f,
     0.5280022621f, -0.3742936850f, -0.5760775208f, 0.7015634775f
);

#if defined(DATA_A_ISO3)
#define BLOCK_BYTE_SIZE 50

uint unpack_iso3_idx(uint8_t qs, uint8_t signs, uint j) {
    uint low = (uint(qs) >> ((j % 4) * 2)) & 0x3;
    uint hi = (uint(signs) >> (j % 8)) & 0x1;
    return low | (hi << 2);
}

FLOAT_TYPEV4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    uint8_t qs = binding_idx == BINDING_IDX_K
        ? k_packed.k_data_packed16[a_offset + ib].qs[iqs / 4]
        : v_packed.v_data_packed16[a_offset + ib].qs[iqs / 4];
    uint8_t signs = binding_idx == BINDING_IDX_K
        ? k_packed.k_data_packed16[a_offset + ib].signs[iqs / 8]
        : v_packed.v_data_packed16[a_offset + ib].signs[iqs / 8];
    float norm = float(binding_idx == BINDING_IDX_K
        ? k_packed.k_data_packed16[a_offset + ib].norm
        : v_packed.v_data_packed16[a_offset + ib].norm);

    int g = int(iqs) / 4;

    float qvals[4];
    for (int c = 0; c < 4; c++) {
        uint idx = unpack_iso3_idx(qs, signs, uint(g * 4 + c));
        qvals[c] = ISO_CENTROIDS_3BIT[idx];
    }

    float qw = ISO_QW[g], qx = -ISO_QX[g], qy = -ISO_QY[g], qz = -ISO_QZ[g];
    float rw = qw*qvals[0] - qx*qvals[1] - qy*qvals[2] - qz*qvals[3];
    float rx = qw*qvals[1] + qx*qvals[0] + qy*qvals[3] - qz*qvals[2];
    float ry = qw*qvals[2] - qx*qvals[3] + qy*qvals[0] + qz*qvals[1];
    float rz = qw*qvals[3] + qx*qvals[2] - qy*qvals[1] + qz*qvals[0];

    return FLOAT_TYPEV4(rw * norm, rx * norm, ry * norm, rz * norm);
}
#endif

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))


// Store column zero. This is used to save per-row m and L values for split_k.
ACC_TYPE perElemOpStoreCol0(const in uint32_t r, const in uint32_t c, const in ACC_TYPE elem, const in uint32_t o_offset, const in uint32_t iq2, const in uint32_t N)
{
    if (r < N && c == 0) {
        uint32_t offset = iq2 + r;
        data_o[o_offset + offset] = D_TYPE(elem);
    }
    return elem;
}

// Load the slope matrix, indexed by Q's dimension 2.
ACC_TYPE perElemOpComputeSlope(const in uint32_t r, const in uint32_t c, const in ACC_TYPE elem, const in uint32_t iq2)
{
    const uint32_t h = iq2 + (r % p.gqa_ratio);

    uint32_t n_head_log2 = p.mask_n_head_log2 & N_LOG2_MASK;

    const ACC_TYPE base = ACC_TYPE(h < n_head_log2 ? p.m0 : p.m1);
    const int      exph = int(h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1);

    return ACC_TYPE(pow(base, ACC_TYPE(exph)));
}

// Load the sink value, indexed by Q's dimension 2.
ACC_TYPE perElemOpGetSink(const in uint32_t r, const in uint32_t c, const in ACC_TYPE elem, const in uint32_t iq2)
{
    const uint32_t h = iq2 + (r % p.gqa_ratio);

    return ACC_TYPE(data_s[h]);
}

uint32_t i, N, KV, split_k_index, Tr, start_j, end_j,
         gqa_iq1, iq2, iq3, rk2, rk3, rv2, rv3, ik2, ik3, iv2, iv3,
         q_stride, k_stride, v_stride, m_stride;

void init_indices()
{
    N = p.N;
    KV = p.KV;

    if (p.k_num > 1) {
        if (p.gqa_ratio > 1) {
            i = 0;
            // batch and split_k share gl_WorkGroupID.x
            gqa_iq1 = gl_WorkGroupID.x / p.k_num;
            split_k_index = gl_WorkGroupID.x % p.k_num;
        } else {
            gqa_iq1 = 0;
            split_k_index = gl_WorkGroupID.x % p.k_num;
            i = gl_WorkGroupID.x / p.k_num;
        }
    } else if (p.gqa_ratio > 1) {
        i = 0;
        gqa_iq1 = gl_WorkGroupID.x;
        split_k_index = 0;
    } else {
        i = gl_WorkGroupID.x;
        gqa_iq1 = 0;
        split_k_index = 0;
    }

    Tr = CEIL_DIV(N, Br);

    start_j = split_k_index * p.split_kv / Bc;
    end_j = CEIL_DIV(min(KV, (split_k_index + 1) * p.split_kv), Bc);

    // When not using grouped query attention, all rows share the same iq2, equal to gl_WorkGroupID.y.
    // When using grouped query attention, each workgroup does gqa_ratio consecutive values of iq2.
    iq2 = gl_WorkGroupID.y * p.gqa_ratio;
    iq3 = gl_WorkGroupID.z;

    // broadcast factors
    rk2 = p.neq2/p.nek2;
    rk3 = p.neq3/p.nek3;

    rv2 = p.neq2/p.nev2;
    rv3 = p.neq3/p.nev3;

    // k indices
    ik3 = iq3 / rk3;
    ik2 = iq2 / rk2;

    // v indices
    iv3 = iq3 / rv3;
    iv2 = iq2 / rv2;

    // nb?1 are already divided by the type size and are in units of elements.
    // When using grouped query attention, Q is indexed by iq2, so the stride
    // should be nb02 (which is in bytes).
    q_stride = p.gqa_ratio > 1 ? (p.nb02 / 4) : p.nb01;
    k_stride = p.nb11;
    v_stride = p.nb21;
    // When using grouped query attention, all rows use the same mask (stride 0).
    // "p.gqa_ratio >> 16" is just a roundabout way of writing zero
    // that prevents the compiler from folding the "&" through the select
    // and breaking the alignment detection.
    m_stride = (p.gqa_ratio > 1) ? (p.gqa_ratio >> 16) : KV;
}

// Bias applied to softmax to stay in fp16 range.
// Based on ggml-cuda issue https://github.com/ggml-org/llama.cpp/issues/18606
const float FATTN_KQ_MAX_OFFSET = 3.0f*0.6931f;

// Store the output when doing grouped query attention.
// Rows index by Q's dimension 2, and the first N rows are valid.
void gqaStore(const in uint32_t r, const in uint32_t c, const in FLOAT_TYPEV4 elems, const in uint32_t o_offset, const in uint32_t iq2, const in uint32_t N)
{
    uint32_t offset = (iq2 + r) * HSV / 4 + c;
    data_ov4[o_offset + offset] = D_TYPEV4(elems);
}
