use tensor_types::{tensor_type, TensorType};

pub struct Params {
    my_param1: i64,
    my_param2: i64,
}
tensor_type!(MyTensor, [my_param1, my_param2], Params, tch::Kind::Double);

fn main() {
    let params = Params {
        my_param1: 2,
        my_param2: 2,
    };

    let tensor = tch::Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]).reshape(vec![2, 2]);
    let _my_tensor = MyTensor::new(tensor, &params).unwrap();
}
