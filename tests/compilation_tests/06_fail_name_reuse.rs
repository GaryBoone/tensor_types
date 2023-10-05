use tensor_types::tensor_type;

pub struct Params {
    my_param1: i64,
}

fn main() {
    tensor_type!(MyTensor, [my_param1], Params, tch::Kind::Float);
    // This line won't compile because the type MyTensor is already defined.
    tensor_type!(MyTensor, [my_param1], Params, tch::Kind::Float);
}
