#include <bits/stdc++.h>

using namespace std;

#define rep(i, from, to) for (int i = from; i < (to); ++i)
#define all(x) x.begin(), x.end()
#define sz(x) (int)(x).size()
typedef long long ll;
typedef pair<int, int> pii;
typedef vector<int> vi;

ll mod = 1e9 + 7;
ll modpow(ll b, ll e) {
    ll ans = 1;
    for (; e; b = b * b % mod, e /= 2)
        if (e & 1)
            ans = ans * b % mod;
    return ans;
}

template <class T, int N> struct Matrix {
    typedef Matrix M;
    array<array<T, N>, N> d{};
    M operator*(const M &m) const {
        M a;
        rep(i, 0, N) rep(j, 0, N) rep(k, 0, N) a.d[i][j] = (a.d[i][j] + d[i][k] * m.d[k][j]) % mod;
        return a;
    }
    vector<T> operator*(const vector<T> &vec) const {
        vector<T> ret(N);
        rep(i, 0, N) rep(j, 0, N) ret[i] += d[i][j] * vec[j];
        return ret;
    }
    M operator^(ll p) const {
        assert(p >= 0);
        M a, b(*this);
        rep(i, 0, N) a.d[i][i] = 1;
        while (p) {
            if (p & 1)
                a = a * b;
            b = b * b;
            p >>= 1;
        }
        return a;
    }
};

vector<ll> berlekampMassey(vector<ll> s) {
    int n = sz(s), L = 0, m = 0;
    vector<ll> C(n), B(n), T;
    C[0] = B[0] = 1;

    ll b = 1;
    rep(i, 0, n) {
        ++m;
        ll d = s[i] % mod;
        rep(j, 1, L + 1) d = (d + C[j] * s[i - j]) % mod;
        if (!d)
            continue;
        T = C;
        ll coef = d * modpow(b, mod - 2) % mod;
        rep(j, m, n) C[j] = (C[j] - coef * B[j - m]) % mod;
        if (2 * L > i)
            continue;
        L = i + 1 - L;
        B = T;
        b = d;
        m = 0;
    }

    C.resize(L + 1);
    C.erase(C.begin());
    for (ll &x : C)
        x = (mod - x) % mod;
    return C;
}
typedef vector<ll> Poly;
ll linearRec(Poly S, Poly tr, ll k) {
    int n = sz(tr);

    auto combine = [&](Poly a, Poly b) {
        Poly res(n * 2 + 1);
        rep(i, 0, n + 1) rep(j, 0, n + 1) res[i + j] = (res[i + j] + a[i] * b[j]) % mod;
        for (int i = 2 * n; i > n; --i)
            rep(j, 0, n) res[i - 1 - j] = (res[i - 1 - j] + res[i] * tr[j]) % mod;
        res.resize(n + 1);
        return res;
    };

    Poly pol(n + 1), e(pol);
    pol[0] = e[1] = 1;

    for (++k; k; k /= 2) {
        if (k % 2)
            pol = combine(pol, e);
        e = combine(e, e);
    }

    ll res = 0;
    rep(i, 0, n) res = (res + pol[i + 1] * S[i]) % mod;
    return res;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    const int SZ = 10;
    ll A = 0, B = 5, K = 1000;
    Matrix<ll, SZ> adj;
    for (int i = 0; i < SZ; i++) {
        for (int j = 0; j < SZ; j++) {
            adj.d[i][j] = rand() % 2;
        }
    }
    vector<ll> cur(SZ);
    cur[A] = 1;
    vector<ll> vals({cur[B]});
    for (int i = 1; i < SZ * 2; i++) {
        cur = adj * cur;
        vals.push_back(cur[B]);
    }
    vector<ll> recurrence = berlekampMassey(vals);
    cout << linearRec(vals, recurrence, K) << endl;
    cout << (adj ^ K).d[B][A] << endl;
}