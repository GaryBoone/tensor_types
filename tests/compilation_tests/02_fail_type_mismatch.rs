use tensor_types::tensor_type;

// This won't compile because the one of the parameters in the Params struct passed-in to the
// tensor_types_ macro can't be converted to an i64. `From<f64>` is not implemented for `i64`.
pub struct Params {
    my_param1: f64,
    my_param2: i64,
}
tensor_type!(MyTensor, [my_param1, my_param2], Params, tch::Kind::Double);

fn main() {}
