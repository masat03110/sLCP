#ifndef MYPROJ_BIGINT_HPP
#define MYPROJ_BIGINT_HPP

#include <cstdint>

struct alignas(64) int192t{
    __uint64_t high;
    __uint128_t low;
    int192t(__uint64_t high, __uint128_t low): high(high), low(low) {}
    int192t(__uint128_t x): high(0), low(x) {}
    int popcount() const {
        return __builtin_popcountll(high) + __builtin_popcountll(low) + __builtin_popcountll(low >> 64);
    }
    // シフト
    int192t operator<<(int n) const {
        if (n < 128){
            return {high << n | (__uint64_t)(low >> (128 - n)), low << n};
        } else {
            return {(__uint64_t)(low << (n - 128)), 0};
        }
    }
    //シフト
    int192t operator>>(int n) const {
        if (n < 128){
            return {high >> n, (low >> n) | ((__uint64_t)high << (128 - n))};
        } else {
            return {0, high >> (n - 128)};
        }
    }
    // OR
    int192t operator|(int192t x) const {
        return {high | x.high, low | x.low};
    }
    // AND
    int192t operator&(int192t x) const {
        return {high & x.high, low & x.low};
    }
    // XOR
    int192t operator^(int192t x) const {
        return {high ^ x.high, low ^ x.low};
    }
    // minus
    int192t operator-(int192t x) const {
        if (low < x.low){
            return {high - x.high - 1, low - x.low};
        } else {
            return {high - x.high, low - x.low};
        }
    }
    // less than
    bool operator<(int192t x) const {
        if (high < x.high) return true;
        if (high > x.high) return false;
        return low < x.low;
    }
    // greater than
    bool operator>(int192t x) const {
        if (high > x.high) return true;
        if (high < x.high) return false;
        return low > x.low;
    }
    //cast to __uint64_t
    operator __uint64_t() const {
        return low;
    }
    // cast to __uint128_t
    operator __uint128_t() const {
        return low;
    }
};

struct alignas(64) int256t{
    __uint128_t high;
    __uint128_t low;
    int256t(__uint128_t high, __uint128_t low): high(high), low(low) {}
    int256t(__uint128_t x): high(0), low(x) {}
    int popcount() const {
        return __builtin_popcountll(high) + __builtin_popcountll(low) + __builtin_popcountll(low >> 64) + __builtin_popcountll(high >> 64);
    }
    // シフト
    int256t operator<<(int n) const {
        if (n < 128){
            return {high << n | (low >> (128 - n)), low << n};
        } else {
            return {low << (n - 128), 0};
        }
    }
    //シフト
    int256t operator>>(int n) const {
        if (n < 128){
            return {high >> n, low >> n | (high << (128 - n))};
        } else {
            return {0, high >> (n - 128)};
        }
    }
    // OR
    int256t operator|(int256t x) const {
        return {high | x.high, low | x.low};
    }
    // AND
    int256t operator&(int256t x) const {
        return {high & x.high, low & x.low};
    }
    // XOR
    int256t operator^(int256t x) const {
        return {high ^ x.high, low ^ x.low};
    }
    // minus
    int256t operator-(int256t x) const {
        if (low < x.low){
            return {high - x.high - 1, low - x.low};
        } else {
            return {high - x.high, low - x.low};
        }
    }
    // less than
    bool operator<(int256t x) const {
        if (high < x.high) return true;
        if (high > x.high) return false;
        return low < x.low;
    }
    // greater than
    bool operator>(int256t x) const {
        if (high > x.high) return true;
        if (high < x.high) return false;
        return low > x.low;
    }
    //cast to __uint64_t
    operator __uint64_t() const {
        return low;
    }
    // cast to __uint128_t
    operator __uint128_t() const {
        return low;
    }
};

struct alignas(64) int320t{
    __uint64_t high;
    int256t low;
    int320t(__uint64_t high, int256t low): high(high), low(low) {}
    int320t(__uint128_t x): high(0), low(x) {}
    int popcount() const {
        return __builtin_popcountll(high) + low.popcount();
    }
    // シフト
    int320t operator<<(int n) const {
        if (n < 256){
            return {high << n | (__uint64_t)(low >> (256 - n)), low << n};
        } else {
            return {low << (n - 256), 0};
        }
    }
    //シフト
    int320t operator>>(int n) const {
        if (n < 256){
            return {high >> n, (low >> n) | ((int256t)high << (256 - n))};
        } else {
            return {0, high >> (n - 256)};
        }
    }
    // OR
    int320t operator|(int320t x) const {
        return {high | x.high, low | x.low};
    }
    // AND
    int320t operator&(int320t x) const {
        return {high & x.high, low & x.low};
    }
    // XOR
    int320t operator^(int320t x) const {
        return {high ^ x.high, low ^ x.low};
    }
    // minus
    int320t operator-(int320t x) const {
        if (low < x.low){
            return {high - x.high - 1, low - x.low};
        } else {
            return {high - x.high, low - x.low};
        }
    }
    // less than
    bool operator<(int320t x) const {
        if (high < x.high) return true;
        if (high > x.high) return false;
        return low < x.low;
    }
    // greater than
    bool operator>(int320t x) const {
        if (high > x.high) return true;
        if (high < x.high) return false;
        return low > x.low;
    }
    //cast to __uint64_t
    operator __uint64_t() const {
        return low;
    }
    // cast to __uint128_t
    operator __uint128_t() const {
        return low;
    }
};

struct alignas(64) int384t{
    __uint128_t high;
    int256t low;
    int384t(__uint128_t high, int256t low): high(high), low(low) {}
    int384t(__uint128_t x): high(0), low(x) {}
    int popcount() const {
        return __builtin_popcountll(high) + __builtin_popcountll(high >> 64) + low.popcount();
    }
    // シフト
    int384t operator<<(int n) const {
        if (n < 256){
            return {high << n | (__uint64_t)(low >> (256 - n)), low << n};
        } else {
            return {low << (n - 256), 0};
        }
    }
    //シフト
    int384t operator>>(int n) const {
        if (n < 256){
            return {high >> n, (low >> n) | ((int256t)high << (256 - n))};
        } else {
            return {0, (int256t)high >> (n - 256)};
        }
    }
    // OR
    int384t operator|(int384t x) const {
        return {high | x.high, low | x.low};
    }
    // AND
    int384t operator&(int384t x) const {
        return {high & x.high, low & x.low};
    }
    // XOR
    int384t operator^(int384t x) const {
        return {high ^ x.high, low ^ x.low};
    }
    // minus
    int384t operator-(int384t x) const {
        if (low < x.low){
            return {high - x.high - 1, low - x.low};
        } else {
            return {high - x.high, low - x.low};
        }
    }
    // less than
    bool operator<(int384t x) const {
        if (high < x.high) return true;
        if (high > x.high) return false;
        return low < x.low;
    }
    // greater than
    bool operator>(int384t x) const {
        if (high > x.high) return true;
        if (high < x.high) return false;
        return low > x.low;
    }
    //cast to __uint64_t
    operator __uint64_t() const {
        return low;
    }
    // cast to __uint128_t
    operator __uint128_t() const {
        return low;
    }
};

struct alignas(64) int448t{
    __uint128_t high;
    int320t low;
    int448t(__uint128_t high, int320t low): high(high), low(low) {}
    int448t(__uint128_t x): high(0), low(x) {}
    int popcount() const {
        return __builtin_popcountll(high) + __builtin_popcountll(high >> 64) + low.popcount();
    }
    // シフト
    int448t operator<<(int n) const {
        if (n < 320){
            return {high << n | (__uint64_t)(low >> (320 - n)), low << n};
        } else {
            return {low << (n - 320), 0};
        }
    }
    //シフト
    int448t operator>>(int n) const {
        if (n < 320){
            return {high >> n, (low >> n) | ((int320t)high << (320 - n))};
        } else {
            return {0, (int320t)high >> (n - 320)};
        }
    }
    // OR
    int448t operator|(int448t x) const {
        return {high | x.high, low | x.low};
    }
    // AND
    int448t operator&(int448t x) const {
        return {high & x.high, low & x.low};
    }
    // XOR
    int448t operator^(int448t x) const {
        return {high ^ x.high, low ^ x.low};
    }
    // minus
    int448t operator-(int448t x) const {
        if (low < x.low){
            return {high - x.high - 1, low - x.low};
        } else {
            return {high - x.high, low - x.low};
        }
    }
    // less than
    bool operator<(int448t x) const {
        if (high < x.high) return true;
        if (high > x.high) return false;
        return low < x.low;
    }
    // greater than
    bool operator>(int448t x) const {
        if (high > x.high) return true;
        if (high < x.high) return false;
        return low > x.low;
    }
    //cast to __uint64_t
    operator __uint64_t() const {
        return low;
    }
    // cast to __uint128_t
    operator __uint128_t() const {
        return low;
    }
};

struct alignas(64) int512t{
    __uint128_t high;
    int384t low;
    int512t(__uint128_t high, int384t low): high(high), low(low) {}
    int512t(__uint128_t x): high(0), low(x) {}
    int popcount() const {
        return __builtin_popcountll(high) + __builtin_popcountll(high >> 64) + low.popcount();
    }
    // シフト
    int512t operator<<(int n) const {
        if (n < 384){
            return {high << n | (__uint128_t)(low >> (384 - n)), low << n};
        } else {
            return {low << (n - 384), 0};
        }
    }
    //シフト
    int512t operator>>(int n) const {
        if (n < 384){
            return {high >> n, (low >> n) | ((int384t)high << (384 - n))};
        } else {
            return {0, (int384t)high >> (n - 384)};
        }
    }
    // OR
    int512t operator|(int512t x) const {
        return {high | x.high, low | x.low};
    }
    // AND
    int512t operator&(int512t x) const {
        return {high & x.high, low & x.low};
    }
    // XOR
    int512t operator^(int512t x) const {
        return {high ^ x.high, low ^ x.low};
    }
    // minus
    int512t operator-(int512t x) const {
        if (low < x.low){
            return {high - x.high - 1, low - x.low};
        } else {
            return {high - x.high, low - x.low};
        }
    }
    // less than
    bool operator<(int512t x) const {
        if (high < x.high) return true;
        if (high > x.high) return false;
        return low < x.low;
    }
    // greater than
    bool operator>(int512t x) const {
        if (high > x.high) return true;
        if (high < x.high) return false;
        return low > x.low;
    }
    //cast to __uint64_t
    operator __uint64_t() const {
        return low;
    }
    // cast to __uint128_t
    operator __uint128_t() const {
        return low;
    }
};

struct alignas(64) int576t{
    __uint128_t high;
    int448t low;
    int576t(__uint128_t high, int448t low): high(high), low(low) {}
    int576t(__uint128_t x): high(0), low(x) {}
    int popcount() const {
        return __builtin_popcountll(high) + __builtin_popcountll(high >> 64) + low.popcount();
    }
    // シフト
    int576t operator<<(int n) const {
        if (n < 448){
            return {high << n | (__uint128_t)(low >> (448 - n)), low << n};
        } else {
            return {low << (n - 448), 0};
        }
    }
    //シフト
    int576t operator>>(int n) const {
        if (n < 448){
            return {high >> n, (low >> n) | ((int448t)high << (448 - n))};
        } else {
            return {0, (int448t)high >> (n - 448)};
        }
    }
    // OR
    int576t operator|(int576t x) const {
        return {high | x.high, low | x.low};
    }
    // AND
    int576t operator&(int576t x) const {
        return {high & x.high, low & x.low};
    }
    // XOR
    int576t operator^(int576t x) const {
        return {high ^ x.high, low ^ x.low};
    }
    // minus
    int576t operator-(int576t x) const {
        if (low < x.low){
            return {high - x.high - 1, low - x.low};
        } else {
            return {high - x.high, low - x.low};
        }
    }
    // less than
    bool operator<(int576t x) const {
        if (high < x.high) return true;
        if (high > x.high) return false;
        return low < x.low;
    }
    // greater than
    bool operator>(int576t x) const {
        if (high > x.high) return true;
        if (high < x.high) return false;
        return low > x.low;
    }
    //cast to __uint64_t
    operator __uint64_t() const {
        return low;
    }
    // cast to __uint128_t
    operator __uint128_t() const {
        return low;
    }
};

struct alignas(64) int640t{
    __uint128_t high;
    int512t low;
    int640t(__uint128_t high, int512t low): high(high), low(low) {}
    int640t(__uint128_t x): high(0), low(x) {}
    int popcount() const {
        return __builtin_popcountll(high) + __builtin_popcountll(high >> 64) + low.popcount();
    }
    // シフト
    int640t operator<<(int n) const {
        if (n < 512){
            return {high << n | (__uint128_t)(low >> (512 - n)), low << n};
        } else {
            return {low << (n - 512), 0};
        }
    }
    //シフト
    int640t operator>>(int n) const {
        if (n < 512){
            return {high >> n, (low >> n) | ((int512t)high << (512 - n))};
        } else {
            return {0, (int512t)high >> (n - 512)};
        }
    }
    // OR
    int640t operator|(int640t x) const {
        return {high | x.high, low | x.low};
    }
    // AND
    int640t operator&(int640t x) const {
        return {high & x.high, low & x.low};
    }
    // XOR
    int640t operator^(int640t x) const {
        return {high ^ x.high, low ^ x.low};
    }
    // minus
    int640t operator-(int640t x) const {
        if (low < x.low){
            return {high - x.high - 1, low - x.low};
        } else {
            return {high - x.high, low - x.low};
        }
    }
    // less than
    bool operator<(int640t x) const {
        if (high < x.high) return true;
        if (high > x.high) return false;
        return low < x.low;
    }
    // greater than
    bool operator>(int640t x) const {
        if (high > x.high) return true;
        if (high < x.high) return false;
        return low > x.low;
    }
    //cast to __uint64_t
    operator __uint64_t() const {
        return low;
    }
    // cast to __uint128_t
    operator __uint128_t() const {
        return low;
    }
};

struct alignas(64) int704t{
    __uint128_t high;
    int576t low;
    int704t(__uint128_t high, int576t low): high(high), low(low) {}
    int704t(__uint128_t x): high(0), low(x) {}
    int popcount() const {
        return __builtin_popcountll(high) + __builtin_popcountll(high >> 64) + low.popcount();
    }
    // シフト
    int704t operator<<(int n) const {
        if (n < 576){
            return {high << n | (__uint128_t)(low >> (576 - n)), low << n};
        } else {
            return {low << (n - 576), 0};
        }
    }
    //シフト
    int704t operator>>(int n) const {
        if (n < 576){
            return {high >> n, (low >> n) | ((int576t)high << (576 - n))};
        } else {
            return {0, (int576t)high >> (n - 576)};
        }
    }
    // OR
    int704t operator|(int704t x) const {
        return {high | x.high, low | x.low};
    }
    // AND
    int704t operator&(int704t x) const {
        return {high & x.high, low & x.low};
    }
    // XOR
    int704t operator^(int704t x) const {
        return {high ^ x.high, low ^ x.low};
    }
    // minus
    int704t operator-(int704t x) const {
        if (low < x.low){
            return {high - x.high - 1, low - x.low};
        } else {
            return {high - x.high, low - x.low};
        }
    }
    // less than
    bool operator<(int704t x) const {
        if (high < x.high) return true;
        if (high > x.high) return false;
        return low < x.low;
    }
    // greater than
    bool operator>(int704t x) const {
        if (high > x.high) return true;
        if (high < x.high) return false;
        return low > x.low;
    }
    //cast to __uint64_t
    operator __uint64_t() const {
        return low;
    }
    // cast to __uint128_t
    operator __uint128_t() const {
        return low;
    }
};

struct alignas(64) int768t{
    __uint128_t high;
    int640t low;
    int768t(__uint128_t high, int640t low): high(high), low(low) {}
    int768t(__uint128_t x): high(0), low(x) {}
    int popcount() const {
        return __builtin_popcountll(high) + __builtin_popcountll(high >> 64) + low.popcount();
    }
    // シフト
    int768t operator<<(int n) const {
        if (n < 640){
            return {high << n | (__uint128_t)(low >> (640 - n)), low << n};
        } else {
            return {low << (n - 640), 0};
        }
    }
    //シフト
    int768t operator>>(int n) const {
        if (n < 640){
            return {high >> n, (low >> n) | ((int640t)high << (640 - n))};
        } else {
            return {0, (int640t)high >> (n - 640)};
        }
    }
    // OR
    int768t operator|(int768t x) const {
        return {high | x.high, low | x.low};
    }
    // AND
    int768t operator&(int768t x) const {
        return {high & x.high, low & x.low};
    }
    // XOR
    int768t operator^(int768t x) const {
        return {high ^ x.high, low ^ x.low};
    }
    // minus
    int768t operator-(int768t x) const {
        if (low < x.low){
            return {high - x.high - 1, low - x.low};
        } else {
            return {high - x.high, low - x.low};
        }
    }
    // less than
    bool operator<(int768t x) const {
        if (high < x.high) return true;
        if (high > x.high) return false;
        return low < x.low;
    }
    // greater than
    bool operator>(int768t x) const {
        if (high > x.high) return true;
        if (high < x.high) return false;
        return low > x.low;
    }
    //cast to __uint64_t
    operator __uint64_t() const {
        return low;
    }
    // cast to __uint128_t
    operator __uint128_t() const {
        return low;
    }
};

struct alignas(64) int832t{
    __uint128_t high;
    int704t low;
    int832t(__uint128_t high, int704t low): high(high), low(low) {}
    int832t(__uint128_t x): high(0), low(x) {}
    int popcount() const {
        return __builtin_popcountll(high) + __builtin_popcountll(high >> 64) + low.popcount();
    }
    // シフト
    int832t operator<<(int n) const {
        if (n < 704){
            return {high << n | (__uint128_t)(low >> (704 - n)), low << n};
        } else {
            return {low << (n - 704), 0};
        }
    }
    //シフト
    int832t operator>>(int n) const {
        if (n < 704){
            return {high >> n, (low >> n) | ((int704t)high << (704 - n))};
        } else {
            return {0, (int704t)high >> (n - 704)};
        }
    }
    // OR
    int832t operator|(int832t x) const {
        return {high | x.high, low | x.low};
    }
    // AND
    int832t operator&(int832t x) const {
        return {high & x.high, low & x.low};
    }
    // XOR
    int832t operator^(int832t x) const {
        return {high ^ x.high, low ^ x.low};
    }
    // minus
    int832t operator-(int832t x) const {
        if (low < x.low){
            return {high - x.high - 1, low - x.low};
        } else {
            return {high - x.high, low - x.low};
        }
    }
    // less than
    bool operator<(int832t x) const {
        if (high < x.high) return true;
        if (high > x.high) return false;
        return low < x.low;
    }
    // greater than
    bool operator>(int832t x) const {
        if (high > x.high) return true;
        if (high < x.high) return false;
        return low > x.low;
    }
    //cast to __uint64_t
    operator __uint64_t() const {
        return low;
    }
    // cast to __uint128_t
    operator __uint128_t() const {
        return low;
    }
};

struct alignas(64) int896t{
    __uint128_t high;
    int768t low;
    int896t(__uint128_t high, int768t low): high(high), low(low) {}
    int896t(__uint128_t x): high(0), low(x) {}
    int popcount() const {
        return __builtin_popcountll(high) + __builtin_popcountll(high >> 64) + low.popcount();
    }
    // シフト
    int896t operator<<(int n) const {
        if (n < 768){
            return {high << n | (__uint128_t)(low >> (768 - n)), low << n};
        } else {
            return {low << (n - 768), 0};
        }
    }
    //シフト
    int896t operator>>(int n) const {
        if (n < 768){
            return {high >> n, (low >> n) | ((int768t)high << (768 - n))};
        } else {
            return {0, (int768t)high >> (n - 768)};
        }
    }
    // OR
    int896t operator|(int896t x) const {
        return {high | x.high, low | x.low};
    }
    // AND
    int896t operator&(int896t x) const {
        return {high & x.high, low & x.low};
    }
    // XOR
    int896t operator^(int896t x) const {
        return {high ^ x.high, low ^ x.low};
    }
    // minus
    int896t operator-(int896t x) const {
        if (low < x.low){
            return {high - x.high - 1, low - x.low};
        } else {
            return {high - x.high, low - x.low};
        }
    }
    // less than
    bool operator<(int896t x) const {
        if (high < x.high) return true;
        if (high > x.high) return false;
        return low < x.low;
    }
    // greater than
    bool operator>(int896t x) const {
        if (high > x.high) return true;
        if (high < x.high) return false;
        return low > x.low;
    }
    //cast to __uint64_t
    operator __uint64_t() const {
        return low;
    }
    // cast to __uint128_t
    operator __uint128_t() const {
        return low;
    }
};

struct alignas(64) int960t{
    __uint128_t high;
    int832t low;
    int960t(__uint128_t high, int832t low): high(high), low(low) {}
    int960t(__uint128_t x): high(0), low(x) {}
    int popcount() const {
        return __builtin_popcountll(high) + __builtin_popcountll(high >> 64) + low.popcount();
    }
    // シフト
    int960t operator<<(int n) const {
        if (n < 832){
            return {high << n | (__uint128_t)(low >> (832 - n)), low << n};
        } else {
            return {low << (n - 832), 0};
        }
    }
    //シフト
    int960t operator>>(int n) const {
        if (n < 832){
            return {high >> n, (low >> n) | ((int832t)high << (832 - n))};
        } else {
            return {0, (int832t)high >> (n - 832)};
        }
    }
    // OR
    int960t operator|(int960t x) const {
        return {high | x.high, low | x.low};
    }
    // AND
    int960t operator&(int960t x) const {
        return {high & x.high, low & x.low};
    }
    // XOR
    int960t operator^(int960t x) const {
        return {high ^ x.high, low ^ x.low};
    }
    // minus
    int960t operator-(int960t x) const {
        if (low < x.low){
            return {high - x.high - 1, low - x.low};
        } else {
            return {high - x.high, low - x.low};
        }
    }
    // less than
    bool operator<(int960t x) const {
        if (high < x.high) return true;
        if (high > x.high) return false;
        return low < x.low;
    }
    // greater than
    bool operator>(int960t x) const {
        if (high > x.high) return true;
        if (high < x.high) return false;
        return low > x.low;
    }
    //cast to __uint64_t
    operator __uint64_t() const {
        return low;
    }
    // cast to __uint128_t
    operator __uint128_t() const {
        return low;
    }
};

struct alignas(64) int1024t{
    __uint128_t high;
    int896t low;
    int1024t(__uint128_t high, int896t low): high(high), low(low) {}
    int1024t(__uint128_t x): high(0), low(x) {}
    int popcount() const {
        return __builtin_popcountll(high) + __builtin_popcountll(high >> 64) + low.popcount();
    }
    // シフト
    int1024t operator<<(int n) const {
        if (n < 896){
            return {high << n | (__uint128_t)(low >> (896 - n)), low << n};
        } else {
            return {low << (n - 896), 0};
        }
    }
    //シフト
    int1024t operator>>(int n) const {
        if (n < 896){
            return {high >> n, (low >> n) | ((int896t)high << (896 - n))};
        } else {
            return {0, (int896t)high >> (n - 896)};
        }
    }
    // OR
    int1024t operator|(int1024t x) const {
        return {high | x.high, low | x.low};
    }
    // AND
    int1024t operator&(int1024t x) const {
        return {high & x.high, low & x.low};
    }
    // XOR
    int1024t operator^(int1024t x) const {
        return {high ^ x.high, low ^ x.low};
    }
    // minus
    int1024t operator-(int1024t x) const {
        if (low < x.low){
            return {high - x.high - 1, low - x.low};
        } else {
            return {high - x.high, low - x.low};
        }
    }
    // less than
    bool operator<(int1024t x) const {
        if (high < x.high) return true;
        if (high > x.high) return false;
        return low < x.low;
    }
    // greater than
    bool operator>(int1024t x) const {
        if (high > x.high) return true;
        if (high < x.high) return false;
        return low > x.low;
    }
    //cast to __uint64_t
    operator __uint64_t() const {
        return low;
    }
    // cast to __uint128_t
    operator __uint128_t() const {
        return low;
    }
};

#endif // MYPROJ_BIGINT_HPP