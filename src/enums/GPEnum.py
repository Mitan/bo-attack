def my_enum(**enums):
    return type('Enum', (), enums)


GPEnum = my_enum(SimpleGP=1,
                 AdditiveGP=2,
                 LearnDimGP=3)
