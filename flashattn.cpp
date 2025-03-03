/*
This code reproduces forward pass of FlashAttention-v1 following Algorithm 1 in the paper.
To enhance readability of codes, I didn't perform code optimization to reduce execution time and reduce memory.
I focused on the concept of tiling and fused operation in the unit of a block. 
*/

# include <iostream>
# include <fstream>
# include <vector>
# include <cmath>
# include <chrono>
# include <random>
# include <algorithm>
# include <filesystem>

using namespace std;

typedef struct EstimatedTime {
    float mem_alloc;
    float qkv_proj;
    float qk_matmul;
    float qk_softmax;
    float qkv_matmul;
    float o_proj;
    float core_time;
    float total_time;
} EstimatedTime;

template<typename T>
class MultiHeadSelfAttention
{   
    public:
        MultiHeadSelfAttention(size_t batch_size, size_t seq_len, size_t num_heads, size_t hidden_dims);
        ~MultiHeadSelfAttention();

        void load_state_dict(T* q, T* k, T* v, T* o, T* qb, T* kb, T* vb, T* ob);
        // T* forward(T* input_tensor);
        // vector<T> forward(T* input_tensor);
        EstimatedTime forward(T* hidden_states);

        size_t _batch_size;
        size_t _seq_len;
        size_t _num_heads;
        size_t _hidden_dims;
        size_t _per_head_hidden_dims;

        T* q_proj_weight;
        T* k_proj_weight;
        T* v_proj_weight;
        T* o_proj_weight;

        T* q_proj_bias;
        T* k_proj_bias;
        T* v_proj_bias;
        T* o_proj_bias;

        float normalizer;
};

template<typename T>
MultiHeadSelfAttention<T>::MultiHeadSelfAttention(size_t batch_size, size_t seq_len, size_t num_heads, size_t hidden_dims)
{
    _batch_size = batch_size;
    _seq_len = seq_len;
    _num_heads = num_heads;
    _hidden_dims = hidden_dims; 
    _per_head_hidden_dims = int(_hidden_dims/_num_heads);

    normalizer = sqrt(float(_per_head_hidden_dims));
};

template<typename T>
MultiHeadSelfAttention<T>::~MultiHeadSelfAttention() 
{
    delete [] q_proj_weight;
    delete [] k_proj_weight;
    delete [] v_proj_weight;
    delete [] o_proj_weight;
};

template<typename T>
void MultiHeadSelfAttention<T>::load_state_dict(T* q, T* k, T* v, T* o, T* qb, T* kb, T* vb, T* ob)
{
    q_proj_weight = q;
    k_proj_weight = k;
    v_proj_weight = v;
    o_proj_weight = o;

    q_proj_bias   = qb;
    k_proj_bias   = kb;
    v_proj_bias   = vb;
    o_proj_bias   = ob;
};

template<typename T>
// T* MultiHeadSelfAttention<T>::forward(T* hidden_states)
EstimatedTime MultiHeadSelfAttention<T>::forward(T* hidden_states)
{
    auto time_stamp_1   = chrono::system_clock::now();

    T* query_states     = new T[_batch_size * _seq_len * _hidden_dims]();
    T* key_states       = new T[_batch_size * _seq_len * _hidden_dims]();
    T* value_states     = new T[_batch_size * _seq_len * _hidden_dims]();
    T* output_states    = new T[_batch_size * _seq_len * _hidden_dims]();

    T* attn_scores      = new T[_batch_size * _num_heads * _seq_len * _seq_len]();
    T* attn_output      = new T[_batch_size * _seq_len * _hidden_dims]();

    auto time_stamp_2   = chrono::system_clock::now();

    // query, key, value projection
    // query/key/value_states(batch_size, seq_len, hidden_dims) =
    //      hidden_states(batch_size, seq_len, hidden_dims) * q/k/v_proj_weight(hidden_dims, hidden_dim)^T
    for (size_t batch_idx = 0; batch_idx < _batch_size; batch_idx++) {
        
        for (size_t seq_idx = 0; seq_idx < _seq_len; seq_idx++) {
            size_t st_vec_idx = batch_idx * _seq_len * _hidden_dims + seq_idx * _hidden_dims;

            for (size_t out_hidden_idx = 0; out_hidden_idx < _hidden_dims; out_hidden_idx++) {
                size_t qkv_st_elem_idx = st_vec_idx + out_hidden_idx;
                size_t w_row_idx = out_hidden_idx * _hidden_dims;

                for (size_t in_hidden_idx = 0; in_hidden_idx < _hidden_dims; in_hidden_idx++) {
                    size_t hidden_st_elem_idx = st_vec_idx + in_hidden_idx; 
                    size_t w_elem_idx = w_row_idx + in_hidden_idx;

                    query_states[qkv_st_elem_idx] += hidden_states[hidden_st_elem_idx] * q_proj_weight[w_elem_idx];
                    key_states[qkv_st_elem_idx] += hidden_states[hidden_st_elem_idx] * k_proj_weight[w_elem_idx];
                    value_states[qkv_st_elem_idx] += hidden_states[hidden_st_elem_idx] * v_proj_weight[w_elem_idx];
                }
                query_states[qkv_st_elem_idx] += q_proj_bias[out_hidden_idx];
                key_states[qkv_st_elem_idx] += k_proj_bias[out_hidden_idx];
                value_states[qkv_st_elem_idx] += v_proj_bias[out_hidden_idx];
            }
        }
    }

    auto time_stamp_3   = chrono::system_clock::now();

    // Assume that L1 cache size is 128[KB] = 2^17[B]; 1 << 17
    // Assume that we are using floating point (FP32) number format; (1 << 17) >> 2;
    // So, L1 cache could allocate memory space for 2^15[FP32].
    // Denoted as `M` of Algorithm 1 in the paper
    size_t cache_size = (1 << 17) >> 2;
    // Denoted as `B_c` of Algorithm 1 in the paper
    size_t col_block_size = ceil(cache_size / (4*_per_head_hidden_dims));
    // Denoted as `B_r` of Algorithm 1 in the paper
    size_t row_block_size = min(int(ceil(cache_size / (4*_per_head_hidden_dims))), int(_per_head_hidden_dims));
    // Denoted as `T_c` of Algorithm 1 in the paper
    size_t num_col_blocks = ceil(_seq_len / col_block_size);
    // Denoted as `T_r` of Algorithm 1 in the paper
    size_t num_row_blocks = ceil(_seq_len / row_block_size);

    for (size_t batch_idx = 0; batch_idx < _batch_size; batch_idx++) {
        size_t st_matrix_offset = batch_idx * _seq_len * _hidden_dims; 
        
        for (size_t head_idx = 0; head_idx < _num_heads; head_idx++) {
            T* row_wise_denominator = new T[_seq_len]();
            T* row_wise_max = new T[_seq_len]; 
            for (size_t seq_idx; seq_idx < _seq_len; seq_idx++) {
                row_wise_max[seq_idx] = -10000.0;
            }

            for (size_t col_blk_idx = 0; col_blk_idx < num_col_blocks; col_blk_idx++) {
                size_t kv_block_size = col_block_size * _per_head_hidden_dims;
                size_t kv_block_offset = st_matrix_offset + col_blk_idx * (col_block_size * _hidden_dims) + head_idx * (_per_head_hidden_dims); 

                // Line 6 of Algorithm 1 in the paper
                // Load K, V block from memory to cache
                T* key_states_block = new T[kv_block_size];
                T* value_states_block = new T[kv_block_size];

                for (size_t col_vec_idx = 0; col_vec_idx < col_block_size; col_vec_idx++){
                    size_t kv_vec_offset = col_vec_idx * _per_head_hidden_dims;
                    memcpy(key_states_block + kv_vec_offset, key_states + kv_block_offset + col_vec_idx * _hidden_dims, _per_head_hidden_dims * sizeof(T));
                    memcpy(value_states_block + kv_vec_offset, value_states + kv_block_offset + col_vec_idx * _hidden_dims, _per_head_hidden_dims * sizeof(T));
                }

                for (size_t row_blk_idx = 0; row_blk_idx < num_row_blocks; row_blk_idx++) {
                    // Line 8 of Algorithm 1 in the paper
                    // Load Q, O blcoks and l, m vector 
                    size_t q_block_size = row_block_size * _per_head_hidden_dims;
                    size_t q_block_offset = st_matrix_offset + row_blk_idx * (row_block_size * _hidden_dims) + head_idx * (_per_head_hidden_dims);

                    T* query_states_block = new T[q_block_size];
                    T* attn_output_block = new T[q_block_size];
                    T* row_wise_denominator_block = new T[row_block_size];
                    T* row_wise_max_block = new T[row_block_size];

                    for (size_t row_vec_idx = 0; row_vec_idx < row_block_size; row_vec_idx++) {
                        size_t q_vec_offset = row_vec_idx * _per_head_hidden_dims;
                        memcpy(query_states_block + q_vec_offset, query_states + q_block_offset + row_vec_idx * _hidden_dims, _per_head_hidden_dims * sizeof(T));
                        memcpy(attn_output_block + q_vec_offset, attn_output + q_block_offset + row_vec_idx * _hidden_dims, _per_head_hidden_dims * sizeof(T));
                    }

                    memcpy(row_wise_denominator_block, row_wise_denominator + row_blk_idx * row_block_size, row_block_size * sizeof(T));
                    memcpy(row_wise_max_block, row_wise_max + row_blk_idx * row_block_size, row_block_size * sizeof(T));

                    // Line 9 of Algorithm 1 in the paper
                    // compute attention_scores (qk block matmul)
                    T* attn_score_block = new T[row_block_size * col_block_size];
                    for (size_t row_vec_idx = 0; row_vec_idx < row_block_size; row_vec_idx++) {
                        
                        for (size_t col_vec_idx = 0; col_vec_idx < col_block_size; col_vec_idx++) {
                        
                            T attn_score_elem = 0;
                            for (size_t hidden_idx = 0; hidden_idx < _per_head_hidden_dims; hidden_idx++) {
                                attn_score_elem += query_states_block[row_vec_idx * _per_head_hidden_dims + hidden_idx] * key_states_block[col_vec_idx * _per_head_hidden_dims + hidden_idx];
                            }
                            attn_score_block[row_vec_idx * col_block_size + col_vec_idx] = attn_score_elem / normalizer;
                        }
                    }


                    // Line 10 of Algorithm 1 in the paper
                    // compute row-wise maximum for the current block and denominator (extra statistics)
                    T* row_wise_max_block_tmp = new T[row_block_size];
                    T* row_wise_denominator_block_tmp = new T[row_block_size];
                    T* attn_prob_block = new T[row_block_size * col_block_size];

                    for (size_t row_vec_idx = 0; row_vec_idx < row_block_size; row_vec_idx++) {
                        size_t row_vec_offset = row_vec_idx * col_block_size;

                        T row_wise_max_elem = *max_element(attn_score_block + row_vec_offset, attn_score_block + row_vec_offset + col_block_size);
                        T row_wise_denominator_elem = 0.0;

                        for (size_t col_vec_idx = 0; col_vec_idx < col_block_size; col_vec_idx++) {
                            attn_prob_block[row_vec_offset + col_vec_idx] = exp(attn_score_block[row_vec_offset + col_vec_idx] - row_wise_max_elem);
                            row_wise_denominator_elem += attn_prob_block[row_vec_offset + col_vec_idx];
                        }
                        row_wise_max_block_tmp[row_vec_idx] = row_wise_max_elem;
                        row_wise_denominator_block_tmp[row_vec_idx] = row_wise_denominator_elem;
                    }

                    // Line 11 of Algorithm 1 in the paper
                    // update extra statistics
                    T* row_wise_max_block_new = new T[row_block_size];
                    T* row_wise_denominator_block_new = new T[row_block_size];
                    for (size_t row_vec_idx = 0; row_vec_idx < row_block_size; row_vec_idx++) {
                        row_wise_max_block_new[row_vec_idx] = max(row_wise_max_block[row_vec_idx], row_wise_max_block_tmp[row_vec_idx]);
                        row_wise_denominator_block_new[row_vec_idx] = exp(row_wise_max_block[row_vec_idx] - row_wise_max_block_new[row_vec_idx]) * row_wise_denominator_block[row_vec_idx];
                        row_wise_denominator_block_new[row_vec_idx] += exp(row_wise_max_block_tmp[row_vec_idx] - row_wise_max_block_new[row_vec_idx]) * row_wise_denominator_block_tmp[row_vec_idx];
                    }

                    // Line 12 of Algorithm 1 in the paper
                    // compute and update `attn_output` with updated extra statistics.
                    // write `attn_output_block` to `attn_output` in memory.
                    for (size_t row_vec_idx = 0; row_vec_idx < row_block_size; row_vec_idx++) {
                        size_t attn_prob_vec_offset = row_vec_idx * col_block_size;
                        size_t attn_output_vec_offset = row_vec_idx * _per_head_hidden_dims;

                        T* attn_output_block_new = new T[_per_head_hidden_dims]();

                        for (size_t hidden_idx = 0; hidden_idx < _per_head_hidden_dims; hidden_idx++) {

                            T attn_output_elem = 0.0;
                            for (size_t col_vec_idx = 0; col_vec_idx < col_block_size; col_vec_idx++) {
                                attn_output_block_new[hidden_idx] += attn_prob_block[attn_prob_vec_offset + col_vec_idx] * value_states_block[col_vec_idx * _per_head_hidden_dims + hidden_idx];
                            }
                            attn_output_elem += row_wise_denominator_block[row_vec_idx] * (row_wise_max_block[row_vec_idx] - row_wise_max_block_new[row_vec_idx]) * attn_output_block[attn_output_vec_offset + hidden_idx];
                            attn_output_elem += exp(row_wise_max_block_tmp[row_vec_idx] - row_wise_max_block_new[row_vec_idx]) * attn_output_block_new[hidden_idx];
                            attn_output_elem /= row_wise_denominator_block_new[row_vec_idx];

                            attn_output_block[attn_output_vec_offset + hidden_idx] = attn_output_elem;
                        }

                        delete [] attn_output_block_new;
                    }

                    for (size_t row_vec_idx = 0; row_vec_idx < row_block_size; row_vec_idx++) {
                        size_t q_vec_offset = row_vec_idx * _per_head_hidden_dims;
                        memcpy(attn_output + q_block_offset + row_vec_idx * _hidden_dims, attn_output_block + q_vec_offset,  _per_head_hidden_dims * sizeof(T));
                    }

                    // Line 13 of Algorithm 1 in the paper
                    // write exra statics to memory
                    memcpy(row_wise_max + row_blk_idx * row_block_size, row_wise_max_block_new, row_block_size * sizeof(T));
                    memcpy(row_wise_denominator + row_blk_idx * row_block_size, row_wise_denominator_block_new, row_block_size * sizeof(T));

                    delete [] attn_score_block;
                    delete [] query_states_block;
                }
                delete [] key_states_block;
                delete [] value_states_block;
            }

            delete [] row_wise_max;
            delete [] row_wise_denominator;
        }
    }

    auto time_stamp_6   = chrono::system_clock::now();

    // output projection
    // output_states(batch_size, seq_len, hidden_dims) =
    //      attn_output(batch_size, seq_len, hidden_dims) * o_proj_weight(hidden_dims, hidden_dims)
    for (size_t batch_idx = 0; batch_idx < _batch_size; batch_idx++) {
        
        for (size_t seq_idx = 0; seq_idx < _seq_len; seq_idx++) {
            size_t st_vec_idx = batch_idx * _seq_len * _hidden_dims + seq_idx * _hidden_dims;

            for (size_t out_hidden_idx = 0; out_hidden_idx < _hidden_dims; out_hidden_idx++) {
                size_t o_st_elem_idx = st_vec_idx + out_hidden_idx;
                size_t w_row_idx = out_hidden_idx * _hidden_dims;

                for (size_t in_hidden_idx = 0; in_hidden_idx < _hidden_dims; in_hidden_idx++) {
                    size_t hidden_st_elem_idx = st_vec_idx + in_hidden_idx;
                    size_t w_elem_idx = w_row_idx + in_hidden_idx;

                    output_states[o_st_elem_idx] += attn_output[hidden_st_elem_idx] * o_proj_weight[w_elem_idx];
                }
                output_states[o_st_elem_idx] += o_proj_bias[out_hidden_idx];
            }
        }
    }

    auto time_stamp_7   = chrono::system_clock::now();

    EstimatedTime est_time;
    est_time.mem_alloc  = chrono::duration_cast<chrono::milliseconds>(time_stamp_2 - time_stamp_1).count();
    est_time.qkv_proj   = chrono::duration_cast<chrono::milliseconds>(time_stamp_3 - time_stamp_2).count();
    // est_time.qk_matmul  = chrono::duration_cast<chrono::milliseconds>(time_stamp_4 - time_stamp_3).count();
    // est_time.qk_softmax = chrono::duration_cast<chrono::milliseconds>(time_stamp_5 - time_stamp_4).count();
    // est_time.qkv_matmul = chrono::duration_cast<chrono::milliseconds>(time_stamp_6 - time_stamp_5).count();
    est_time.o_proj     = chrono::duration_cast<chrono::milliseconds>(time_stamp_7 - time_stamp_6).count();
    est_time.total_time = chrono::duration_cast<chrono::milliseconds>(time_stamp_7 - time_stamp_1).count();
    est_time.core_time = chrono::duration_cast<chrono::milliseconds>(time_stamp_6 - time_stamp_3).count();

    cout << "total time: " << est_time.total_time << endl;

    // return output_states;
    return est_time;
};

void read_binary_file(string file_name, float*& tensor)
{
    // open the file
    std::streampos file_size;
    std::ifstream file(file_name, std::ios::binary);

    // get its size
    file.seekg(0, std::ios::end);
    file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    // read the data
    file.read((char*) tensor, file_size);
};

int main()
{
    bool use_random_value   = false;

    size_t batch_size       = 2;
    size_t seq_len          = 128;
    size_t num_heads        = 12;
    size_t hidden_dims      = 768;

    size_t num_samples = 5;

    float* hidden_states    = new float[batch_size * seq_len * hidden_dims];

    float* q_proj_weight    = new float[hidden_dims * hidden_dims];
    float* k_proj_weight    = new float[hidden_dims * hidden_dims];
    float* v_proj_weight    = new float[hidden_dims * hidden_dims];
    float* o_proj_weight    = new float[hidden_dims * hidden_dims];

    float* q_proj_bias      = new float[hidden_dims];
    float* k_proj_bias      = new float[hidden_dims];
    float* v_proj_bias      = new float[hidden_dims];
    float* o_proj_bias      = new float[hidden_dims];

    if (use_random_value) {
        std::mt19937 prng(std::random_device{}());
        std::uniform_real_distribution<float> dist(-1.0, 1.0);

        // assign weights with random float numbers from uniform distribution ranging from -1 to 1
        for (size_t idx = 0; idx < hidden_dims * hidden_dims; idx++) {
            q_proj_weight[idx] = dist(prng);
            k_proj_weight[idx] = dist(prng);
            v_proj_weight[idx] = dist(prng);
            o_proj_weight[idx] = dist(prng);
        }

        for (size_t idx = 0; idx < hidden_dims; idx++) {
            q_proj_bias[idx] = dist(prng);
            k_proj_bias[idx] = dist(prng);
            v_proj_bias[idx] = dist(prng);
            o_proj_bias[idx] = dist(prng);
        }

        // assign hidden_states with random float numbers from uniform distribution ranging from -1 to 1
        for (size_t idx = 0; idx < batch_size * seq_len * hidden_dims; idx++) {
            hidden_states[idx] = dist(prng);
        }
    }
    else {
        batch_size       = 2;
        seq_len          = 128;
        num_heads        = 12;
        hidden_dims      = 768;

        num_samples = 1;

        filesystem::path cwd = filesystem::current_path();

        // model from HuggingFace hub. google-bert/bert-base-uncased.
        // hidden_states and other state tensors with (batch_size, seq_len, hidden_dims) = (2,128,768)
        read_binary_file((cwd / "..\\bert-base-uncased\\query.weight").generic_string(), q_proj_weight);
        read_binary_file((cwd / "..\\bert-base-uncased\\key.weight").generic_string(), k_proj_weight);
        read_binary_file((cwd / "..\\bert-base-uncased\\value.weight").generic_string(), v_proj_weight);
        read_binary_file((cwd / "..\\bert-base-uncased\\output.weight").generic_string(), o_proj_weight);

        read_binary_file((cwd / "..\\bert-base-uncased\\query.bias").generic_string(), q_proj_bias);
        read_binary_file((cwd / "..\\bert-base-uncased\\key.bias").generic_string(), k_proj_bias);
        read_binary_file((cwd / "..\\bert-base-uncased\\value.bias").generic_string(), v_proj_bias);
        read_binary_file((cwd / "..\\bert-base-uncased\\output.bias").generic_string(), o_proj_bias);

        read_binary_file((cwd / "..\\bert-base-uncased\\hidden_states").generic_string(), hidden_states);
    }

    MultiHeadSelfAttention<float> attn(batch_size, seq_len, num_heads, hidden_dims);
    attn.load_state_dict(q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight, q_proj_bias, k_proj_bias, v_proj_bias, o_proj_bias);
    
    // float* output_states = attn.forward(hidden_states);

    EstimatedTime est_time[num_samples];
    for (size_t idx = 0; idx < num_samples; idx++) {
        // float* output_states = attn.forward(hidden_states);
        est_time[idx] = attn.forward(hidden_states);
    }

    return 0;
}