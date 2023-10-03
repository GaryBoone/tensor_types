use tensor_types::{parameter_type, tensor_type};

parameter_type!(BatchSize, i64);
parameter_type!(SequenceLength, i64);
tensor_type!(InputTensor, [BatchSize, SequenceLength], tch::Kind::Float);

fn main() {
    // This won't compile because the passed-in types don't match the declared types. They're
    // reversed.
    InputTensor::set(SequenceLength(2), BatchSize(2)).unwrap();
}
