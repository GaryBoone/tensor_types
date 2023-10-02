use tensor_types::{parameter_type, tensor_type};

parameter_type!(BatchSize, i64);
parameter_type!(SequenceLength, i64);
parameter_type!(ModelDimension, i64);
tensor_type!(BatchSeqTensor, [BatchSize, SequenceLength]);
tensor_type!(
    BatchSeqModelTensor,
    [BatchSize, SequenceLength, ModelDimension]
);

fn transform(in1: &BatchSeqTensor, in2: &BatchSeqModelTensor) {
    in1.tensor();
    in2.tensor();
}

fn main() {
    BatchSeqTensor::set(BatchSize(1), SequenceLength(2)).unwrap();
    BatchSeqModelTensor::set(BatchSize(1), SequenceLength(2), ModelDimension(3)).unwrap();

    let t1 = tch::Tensor::from_slice(&[1, 2]).reshape(vec![1, 2]);
    let bst_in = BatchSeqTensor::new(t1).unwrap();
    let t2 = tch::Tensor::from_slice(&[1, 2, 3, 4, 5, 6]).reshape(vec![1, 2, 3]);
    let bsdt_in = BatchSeqModelTensor::new(t2).unwrap();

    // This won't compile because the passed-in types don't match the declared types. They're
    // reversed.
    transform(&bsdt_in, &bst_in);
}
