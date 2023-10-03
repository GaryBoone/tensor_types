use tensor_types::tensor_type;

tensor_type!(MyTensor, [i64, i64], tch::Kind::Float);

fn main() {
    // This won't compile because the passed-in types don't match the declared types.
    MyTensor::set(2i32, 2i64);
}
