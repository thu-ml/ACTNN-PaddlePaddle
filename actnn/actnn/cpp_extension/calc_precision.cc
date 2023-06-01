#include <paddle/extension.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <queue>
// Greedy algorithm
paddle::Tensor calc_precision(paddle::Tensor b, paddle::Tensor C, paddle::Tensor w, double target) {
    PADDLE_ENFORCE_EQ(b.place(), paddle::PlaceType::kCPU, "b must be a CPU tensor!");
    PADDLE_ENFORCE(b.is_contiguous(), "b must be contiguous!");
    PADDLE_ENFORCE_EQ(C.place(), paddle::PlaceType::kCPU, "C must be a CPU tensor!");
    PADDLE_ENFORCE(C.is_contiguous(), "C must be contiguous!");
    PADDLE_ENFORCE_EQ(w.place(), paddle::PlaceType::kCPU, "w must be a CPU tensor!");
    PADDLE_ENFORCE(w.is_contiguous(), "w must be contiguous!");
    // min \sum_i C_i / (2^b_i - 1)^2, s.t., \sum_i b_i = N b
    std::priority_queue<std::pair<float, int64_t>> q;
    auto *b_data = b.data<int>();
    auto *C_data = C.data<float>();
    auto *w_data = w.data<int>();
    auto get_obj = [&](float C, int b) {
        int coeff_1 = ((1 << b) - 1) * ((1 << b) - 1);
        int coeff_2 = ((1 << (b-1)) - 1) * ((1 << (b-1)) - 1);
        return C * (1.0 / coeff_1 - 1.0 / coeff_2);     // negative
    };
    int64_t N = b.size(0);
    double b_sum = 0;
    for (int64_t i = 0; i < N; i++) {
        auto delta = get_obj(C_data[i], b_data[i]) / w_data[i];
        q.push(std::make_pair(delta, i));
        b_sum += b_data[i] * w_data[i];
    }
    while (b_sum > target) {        // Pick up the smallest increment (largest decrement)
        PADDLE_ENFORCE(!q.empty());
        auto i = q.top().second;
        q.pop();
        b_data[i] -= 1;
        b_sum -= w_data[i];
        if (b_data[i] > 1) {
            auto delta = get_obj(C_data[i], b_data[i]) / w_data[i];
            q.push(std::make_pair(delta, i));
        }
    }
    return b;
}
struct State {
    float obj;
    int64_t p, b;
};

// Dynamic programming
std::pair<paddle::Tensor, paddle::Tensor> calc_precision_dp(paddle::Tensor A, paddle::Tensor C, int max_b, int target, int states) {
    using namespace std;
    PADDLE_ENFORCE_EQ(A.place(), paddle::PlaceType::kCPU, "A must be a CPU tensor!");
    PADDLE_ENFORCE_EQ(A.is_contiguous(), true, "A must be contiguous!");
    PADDLE_ENFORCE_EQ(C.place(), paddle::PlaceType::kCPU, "C must be a CPU tensor!");
    PADDLE_ENFORCE_EQ(C.is_contiguous(), true, "C must be contiguous!");
    int64_t N  = A.size(0);
    auto *A_data = A.data<float>();
    auto *C_data = C.data<float>();
    int64_t total_states = target * N * states;
    // Initialize
    std::vector<std::vector<State>> f(N+1);
    for (auto &v: f) {
        v.resize(total_states + 1);
        for (auto &state: v)
            state.obj = 1e20;
    }
    f[0][0].obj = 0;
//    cout << "Initialized " << total_states << endl;
    for (int64_t i = 1; i <= N; i++) {
        // Moving from f[i-1] to f[i]
        for (int64_t b = 0; b < total_states; b++) {
            auto &old_state = f[i-1][b];

            for (int64_t b0 = 1; b0 <= max_b; b0++)
                for (int64_t p = 1; p <= states; p++)
                    if (b + b0 * p <= total_states) {
                        auto &new_state = f[i][b + b0 * p];
                        float p0 = (float)p / states;
                        float B = (1<<b0) - 1;
                        auto delta = (1 - p0) / p0 * A_data[i-1] + C_data[i-1] / (p0 * B * B);
                        if (old_state.obj + delta < new_state.obj) {
                            new_state.obj = old_state.obj + delta;
                            new_state.p = p;
                            new_state.b = b0;
                        }
                    }
        }
    }
//    cout << "DP Finished " << f[N][total_states].obj << endl;
    // Backtrace
    auto b_vec = paddle::zeros({N}, A.place());
    auto p_vec = paddle::zeros({N}, A.place());
    int64_t current_state = total_states;
    for (int64_t i = N; i > 0; i--) {
        auto &state = f[i][current_state];
        b_vec[i-1] = state.b;
        p_vec[i-1] = (float)state.p / states;
        current_state -= state.b * state.p;
    }
    PADDLE_ENFORCE_EQ(current_state, 0, "DP Failed: no path to initial state!");
    return std::make_pair(b_vec, p_vec);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("calc_precision", &calc_precision, "calc_precision");
  m.def("calc_precision_dp", &calc_precision_dp, "calc_precision_dp");
}