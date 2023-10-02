use trybuild::TestCases;

#[test]
// These tests illustrate how the ParameterTypes and TensorTypes catch type mismatches at compile
// time.
fn compilation_tests() {
    let t = TestCases::new();
    t.pass("tests/compilation_tests/01_basic.rs");
    t.compile_fail("tests/compilation_tests/02_fail_type_mismatch.rs");
    t.compile_fail("tests/compilation_tests/03_fail_type_mismatch2.rs");
    t.compile_fail("tests/compilation_tests/04_fail_fn_parameters.rs");
    t.compile_fail("tests/compilation_tests/05_fail_into_inner.rs");
    t.compile_fail("tests/compilation_tests/06_fail_missing_into.rs");
    t.compile_fail("tests/compilation_tests/07_fail_name_reuse.rs");
}
