use tch::Tensor;
use tensor_types::tensor_type;

tensor_type!(MyTensor, [i64, i64], Kind::Double);

fn main() {
    MyTensor::set(2, 2).unwrap();
    let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]).reshape(vec![2, 2]);
    let _my_tensor = MyTensor::new(tensor).unwrap();
}
