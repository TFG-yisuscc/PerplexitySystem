#pragma once
// Minimal public-domain SHA-256 (FIPS 180-4). Single-header, no dependencies.
#include <array>
#include <cstdint>
#include <cstring>
#include <string>

namespace sha256_impl {

static constexpr uint32_t K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
};

inline uint32_t rotr32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

struct Hasher {
    uint32_t h[8] = {
        0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
        0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19,
    };
    uint64_t total_bytes = 0;
    uint8_t  buf[64]     = {};
    int      buf_len     = 0;

    void compress(const uint8_t* blk) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (uint32_t(blk[i*4])   << 24) | (uint32_t(blk[i*4+1]) << 16) |
                   (uint32_t(blk[i*4+2]) <<  8) |  uint32_t(blk[i*4+3]);
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = rotr32(w[i-15], 7) ^ rotr32(w[i-15],18) ^ (w[i-15] >> 3);
            uint32_t s1 = rotr32(w[i- 2],17) ^ rotr32(w[i- 2],19) ^ (w[i- 2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hv=h[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t S1  = rotr32(e,6)^rotr32(e,11)^rotr32(e,25);
            uint32_t ch  = (e&f)^(~e&g);
            uint32_t t1  = hv + S1 + ch + K[i] + w[i];
            uint32_t S0  = rotr32(a,2)^rotr32(a,13)^rotr32(a,22);
            uint32_t maj = (a&b)^(a&c)^(b&c);
            uint32_t t2  = S0 + maj;
            hv=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hv;
    }

    void update(const uint8_t* data, size_t len) {
        total_bytes += len;
        while (len > 0) {
            int take = (64 - buf_len < (int)len) ? (64 - buf_len) : (int)len;
            std::memcpy(buf + buf_len, data, take);
            buf_len += take;
            data    += take;
            len     -= take;
            if (buf_len == 64) { compress(buf); buf_len = 0; }
        }
    }

    std::array<uint8_t,32> digest() {
        Hasher cp = *this;
        uint64_t orig_bits = cp.total_bytes * 8;

        // Append 0x80 padding bit
        uint8_t pad = 0x80;
        cp.buf[cp.buf_len++] = pad;
        if (cp.buf_len > 56) {
            while (cp.buf_len < 64) cp.buf[cp.buf_len++] = 0;
            cp.compress(cp.buf);
            cp.buf_len = 0;
        }
        while (cp.buf_len < 56) cp.buf[cp.buf_len++] = 0;

        // Append 64-bit big-endian bit count
        for (int i = 0; i < 8; ++i)
            cp.buf[56+i] = (orig_bits >> (56 - 8*i)) & 0xff;
        cp.compress(cp.buf);

        std::array<uint8_t,32> out{};
        for (int i = 0; i < 8; ++i) {
            out[i*4+0] = (cp.h[i] >> 24) & 0xff;
            out[i*4+1] = (cp.h[i] >> 16) & 0xff;
            out[i*4+2] = (cp.h[i] >>  8) & 0xff;
            out[i*4+3] =  cp.h[i]        & 0xff;
        }
        return out;
    }
};

// Returns lowercase hex SHA-256 of the given byte buffer.
inline std::string hash_bytes(const char* data, size_t len) {
    Hasher h;
    h.update(reinterpret_cast<const uint8_t*>(data), len);
    auto d = h.digest();
    static const char hex[] = "0123456789abcdef";
    std::string out(64, '0');
    for (int i = 0; i < 32; ++i) {
        out[i*2]   = hex[(d[i] >> 4) & 0xf];
        out[i*2+1] = hex[ d[i]       & 0xf];
    }
    return out;
}

} // namespace sha256_impl
