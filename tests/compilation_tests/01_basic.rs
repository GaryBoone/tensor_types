use tensor_types::tensor_type;

tensor_type!(MyTensor, [i64, i64]);

fn main() {
    MyTensor::set(2, 2).unwrap();
    let tensor = tch::Tensor::from_slice(&[1, 2, 3, 4]).reshape(vec![2, 2]);
    let _my_tensor = MyTensor::new(tensor).unwrap();
}
