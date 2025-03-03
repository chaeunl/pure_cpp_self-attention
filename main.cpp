/*
I focused on the followings:
1. vector caching.
2. fuse softmax operation with attn_scores computation.
Vector caching is effective in all cases. 
Though it cannot lead to reduction in the order of time complexity, it can reduce execution time linearly.
By fusing softmax with attention scores operations, one can avoid main memory access.
*/
# include <iostream>
# include <fstream>
# include <vector>
# include <cmath>
# include <chrono>
# include <random>
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

            // states vector is accessed by O(_hidden_dims) times
            // , so that it would increase memory access time if it is allocated in the main memory.  
            T* hidden_states_cache = new T[_hidden_dims];
            T* query_states_cache = new T[_hidden_dims];
            T* key_states_cache = new T[_hidden_dims];
            T* value_states_cache = new T[_hidden_dims];

            // To increase cache hit rate, declare array to store vector of each state
            // , which can reduce memory access time.
            // I mainly focus on caching vectors without tiling because tiling ratio should be determined carefully
            // , considering available cache and memory in CPU.
            memcpy(hidden_states_cache, hidden_states + st_vec_idx, _hidden_dims  * sizeof(T));
            memcpy(query_states_cache, query_states + st_vec_idx, _hidden_dims  * sizeof(T));
            memcpy(key_states_cache, key_states + st_vec_idx, _hidden_dims  * sizeof(T));
            memcpy(value_states_cache, value_states + st_vec_idx, _hidden_dims  * sizeof(T));

            for (size_t out_hidden_idx = 0; out_hidden_idx < _hidden_dims; out_hidden_idx++) {
                size_t w_row_idx = out_hidden_idx * _hidden_dims;
                
                // for the weight vector, allocating cache for weight vector is harmful
                // because the size of weight vector leads to frequent memory access than cache optimization in C++ compiler
                T* q_proj_weight_cache = q_proj_weight + w_row_idx;
                T* k_proj_weight_cache = k_proj_weight + w_row_idx;
                T* v_proj_weight_cache = v_proj_weight + w_row_idx;

                for (size_t in_hidden_idx = 0; in_hidden_idx < _hidden_dims; in_hidden_idx++) {
                    query_states_cache[out_hidden_idx] += hidden_states_cache[in_hidden_idx] * q_proj_weight_cache[in_hidden_idx];
                    key_states_cache[out_hidden_idx] += hidden_states_cache[in_hidden_idx] * k_proj_weight_cache[in_hidden_idx];
                    value_states_cache[out_hidden_idx] += hidden_states_cache[in_hidden_idx] * v_proj_weight_cache[in_hidden_idx];

                }
                query_states_cache[out_hidden_idx] += q_proj_bias[out_hidden_idx];
                key_states_cache[out_hidden_idx] += k_proj_bias[out_hidden_idx];
                value_states_cache[out_hidden_idx] += v_proj_bias[out_hidden_idx];

            }
            // return computed states from cache to main memory (or maybe cache in the higher hierarchy)
            memcpy(query_states + st_vec_idx, query_states_cache, _hidden_dims  * sizeof(T));
            memcpy(key_states + st_vec_idx, key_states_cache, _hidden_dims  * sizeof(T));
            memcpy(value_states + st_vec_idx, value_states_cache, _hidden_dims  * sizeof(T));

            // free cache
            delete [] hidden_states_cache; 
            delete [] query_states_cache;
            delete [] key_states_cache;
            delete [] value_states_cache;
        }
    }

    auto time_stamp_3   = chrono::system_clock::now();

    // compute attention_scores & compute softmax for `attn_scores` along row-wise axis
    // attn_scores(batch_size, num_heads, seq_len, seq_len) =  
    //      query_states(batch_size, seq_len, num_heads, per_head_hidden_dims) * key_states(batch_size, seq_len)^T / sqrt(per_head_hidden_dims)
    for (size_t batch_idx = 0; batch_idx < _batch_size; batch_idx++) {
        size_t q_tensor_idx = batch_idx * (_num_heads * _seq_len * _per_head_hidden_dims);
        size_t k_tensor_idx = batch_idx * (_num_heads * _seq_len * _per_head_hidden_dims);

        for (size_t head_idx = 0; head_idx < _num_heads; head_idx++) {
            size_t as_tensor_idx = batch_idx * (_num_heads * _seq_len * _seq_len) + head_idx * (_seq_len * _seq_len);

            for (size_t q_seq_idx = 0; q_seq_idx < _seq_len; q_seq_idx++) {
                T denominator = 0.0;
                T exp_as_elem = 0.0;

                size_t as_row_idx = as_tensor_idx + q_seq_idx * _seq_len;
                size_t q_row_idx = q_tensor_idx + q_seq_idx * (_hidden_dims) + head_idx * (_per_head_hidden_dims);

                T* attn_scores_cache = new T[_seq_len];
                T* query_states_cache = new T[_hidden_dims];

                // In the similar context as mentioned above, vectors of attention scores and query states are allocated in the cache.
                memcpy(attn_scores_cache, attn_scores + as_row_idx, _seq_len * sizeof(T));
                memcpy(query_states_cache, query_states + q_row_idx, _hidden_dims * sizeof(T));

                for (size_t k_seq_idx = 0; k_seq_idx < _seq_len; k_seq_idx++){
                    size_t k_row_idx = k_tensor_idx + k_seq_idx * (_hidden_dims) + head_idx * (_per_head_hidden_dims);

                    T* key_states_cache = key_states + k_row_idx;

                    for (size_t hidden_idx = 0; hidden_idx < _per_head_hidden_dims; hidden_idx++) {
                        attn_scores_cache[k_seq_idx] += (query_states_cache[hidden_idx] * key_states_cache[hidden_idx]);
                    }
                    attn_scores_cache[k_seq_idx] /= normalizer;

                    // to compute softmax, exp(*) over attention score in place.
                    // I didn't apply technique for numerical stability of exp(*) here 
                    exp_as_elem = exp(attn_scores_cache[k_seq_idx]);
                    denominator += exp_as_elem;
                    attn_scores_cache[k_seq_idx] = exp_as_elem;
                }

                delete [] query_states_cache;

                for (size_t k_seq_idx = 0; k_seq_idx < _seq_len; k_seq_idx++) {
                    attn_scores_cache[k_seq_idx] = attn_scores_cache[k_seq_idx] / denominator;
                }

                memcpy(attn_scores + as_row_idx, attn_scores_cache, _seq_len * sizeof(T));
                delete [] attn_scores_cache;
            }
        }
    }

    auto time_stamp_4   = chrono::system_clock::now();

    // compute softmax for `attn_scores` along row-wise axis
    // it is merged in to the above computation.

    auto time_stamp_5   = chrono::system_clock::now();

    // multiply attn_scores with value_states
    // attn_outputs(batch_size, seq_len, hidden_dims) = 
    //      attn_scores(batch_size, num_heads, seq_len, seq_len) * value_states(batch_size, num_heads, seq_len, per_head_hidden_dims)
    for (size_t batch_idx = 0; batch_idx < _batch_size; batch_idx++) {
        
        for (size_t head_idx = 0; head_idx < _num_heads; head_idx++) {
            size_t as_tensor_idx = batch_idx * _num_heads * _seq_len * _seq_len + head_idx * _seq_len *_seq_len;

            for (size_t s_seq_idx = 0; s_seq_idx < _seq_len; s_seq_idx++) {
                size_t as_row_idx = as_tensor_idx + s_seq_idx * _seq_len;
                size_t o_row_idx = batch_idx * _seq_len * _hidden_dims + s_seq_idx * _hidden_dims + head_idx * _per_head_hidden_dims;

                T* attn_scores_cache = new T[_seq_len];
                T* attn_output_cache = new T[_hidden_dims];

                memcpy(attn_scores_cache, attn_scores + as_row_idx, _seq_len * sizeof(T));
                memcpy(attn_output_cache, attn_output + o_row_idx, _hidden_dims * sizeof(T));

                for (size_t hidden_idx = 0; hidden_idx < _per_head_hidden_dims; hidden_idx++) {
                    size_t v_col_idx = batch_idx * _seq_len * _hidden_dims + head_idx * _per_head_hidden_dims + hidden_idx;
            
                    T* value_states_cache = value_states + v_col_idx;

                    for (size_t v_seq_idx = 0; v_seq_idx < _seq_len; v_seq_idx++) {
                        attn_output_cache[hidden_idx] += attn_scores_cache[v_seq_idx] * value_states_cache[v_seq_idx * _hidden_dims];
                    }
                }

                memcpy(attn_output + o_row_idx, attn_output_cache, _hidden_dims * sizeof(T));

                delete [] attn_scores_cache;
                delete [] attn_output_cache;
            }
        }
    }

    // concatenate split tensor over multi-head for `attn_output`
    // but, we can skip this operation because of the memory structure for `attn_output`

    auto time_stamp_6   = chrono::system_clock::now();

    // output projection
    // output_states(batch_size, seq_len, hidden_dims) =
    //      attn_output(batch_size, seq_len, hidden_dims) * o_proj_weight(hidden_dims, hidden_dims)
    for (size_t batch_idx = 0; batch_idx < _batch_size; batch_idx++) {
        
        for (size_t seq_idx = 0; seq_idx < _seq_len; seq_idx++) {
            size_t st_vec_idx = batch_idx * _seq_len * _hidden_dims + seq_idx * _hidden_dims;

            T* output_states_cache = new T[_hidden_dims];
            T* attn_output_cache = new T[_hidden_dims];

            memcpy(output_states_cache, output_states + st_vec_idx, _hidden_dims * sizeof(T));
            memcpy(attn_output_cache, attn_output + st_vec_idx, _hidden_dims * sizeof(T));

            for (size_t out_hidden_idx = 0; out_hidden_idx < _hidden_dims; out_hidden_idx++) {
                size_t w_row_idx = out_hidden_idx * _hidden_dims;

                T* o_proj_weight_cache = o_proj_weight + w_row_idx;

                for (size_t in_hidden_idx = 0; in_hidden_idx < _hidden_dims; in_hidden_idx++) {
                    output_states_cache[out_hidden_idx] += attn_output_cache[in_hidden_idx] * o_proj_weight_cache[in_hidden_idx];
                }
                output_states_cache[out_hidden_idx] += o_proj_bias[out_hidden_idx];
            }
            memcpy(output_states + st_vec_idx, output_states_cache, _hidden_dims * sizeof(T));

            delete [] output_states_cache;
            delete [] attn_output_cache;
        }
    }

    auto time_stamp_7   = chrono::system_clock::now();

    EstimatedTime est_time;
    est_time.mem_alloc  = chrono::duration_cast<chrono::milliseconds>(time_stamp_2 - time_stamp_1).count();
    est_time.qkv_proj   = chrono::duration_cast<chrono::milliseconds>(time_stamp_3 - time_stamp_2).count();
    est_time.qk_matmul  = chrono::duration_cast<chrono::milliseconds>(time_stamp_4 - time_stamp_3).count();
    est_time.qk_softmax = chrono::duration_cast<chrono::milliseconds>(time_stamp_5 - time_stamp_4).count();
    est_time.qkv_matmul = chrono::duration_cast<chrono::milliseconds>(time_stamp_6 - time_stamp_5).count();
    est_time.o_proj     = chrono::duration_cast<chrono::milliseconds>(time_stamp_7 - time_stamp_6).count();
    est_time.total_time = chrono::duration_cast<chrono::milliseconds>(time_stamp_7 - time_stamp_2).count();
    est_time.core_time = chrono::duration_cast<chrono::milliseconds>(time_stamp_6 - time_stamp_3).count();

    cout << "total time: " << est_time.total_time << endl;

    delete [] query_states;
    delete [] key_states;
    delete [] value_states;
    delete [] output_states;

    delete [] attn_output;
    delete [] attn_scores;

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
    // if true, then all state tensors and parameters are initialized with values from U(-1,1)
    // otherwise, load `hidden_states` and parameters that are hooked from HuggingFace google-bert/bert-base-uncased model.
    // batch_size, seq_len, num_heads, and hidden_dims are determined in accordance with stored data.  
    bool use_random_value   = false;

    size_t batch_size       = 2;
    size_t seq_len          = 128;
    size_t num_heads        = 12;
    size_t hidden_dims      = 768;

    size_t num_samples = 5;

    // for contiguous memory, all tensors are flattened into 1d array.
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