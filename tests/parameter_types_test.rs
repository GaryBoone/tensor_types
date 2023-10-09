#[cfg(test)]
mod tests {
    use tensor_types::parameter_type;

    parameter_type!(TestParamType, i64);

    #[test]
    fn test_from() {
        let value = TestParamType::from(42);
        assert_eq!(*value, 42);
    }

    #[test]
    fn test_deref() {
        let value: i64 = 5;
        let new_type_value = TestParamType(value);
        assert_eq!(*new_type_value, value);
    }

    #[test]
    fn test_deref_mut() {
        let mut value = TestParamType(42);
        *value = 43;
        assert_eq!(*value, 43);
    }

    #[test]
    fn test_display() {
        let value = TestParamType::from(42);
        assert_eq!(format!("{}", value), "42");
    }

    #[test]
    fn test_debug() {
        let value = TestParamType::from(42);
        assert_eq!(format!("{:?}", value), "TestParamType(42)");
    }

    #[test]
    fn test_serde_serialize() {
        let value = TestParamType::from(42);
        assert_eq!(serde_json::to_string(&value).unwrap(), "42");
    }

    #[test]
    fn test_serde_deserialize() {
        let value: TestParamType = serde_json::from_str("42").unwrap();
        assert_eq!(*value, 42);
    }
}
