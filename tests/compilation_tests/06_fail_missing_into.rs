use tensor_types::tensor_type;

fn main() {
    // The goal is to create a tensor type that can be set to a value of type MyStruct.
    #[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Hash)]
    pub struct MyStruct(i64);

    // The problem is that the user has not implemented From<MyStruct> for i64.
    //
    // It would look like:
    // impl From<MyStruct> for i64 {
    //     fn from(item: MyStruct) -> i64 {
    //         item.0
    //     }
    // }

    tensor_type!(MyTensorStruct, [MyStruct], tch::Kind::Float);
}
