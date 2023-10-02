use tensor_types::tensor_type;

fn main() {
    tensor_type!(MyTensor, [i64]);
    // This line won't compile because the type MyTensor is already defined.
    tensor_type!(MyTensor, [i64, i64]);
}
