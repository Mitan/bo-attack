def enum(**enums):
    return type('Enum', (), enums)


GPEnum = enum(SimpleGP=1,
              AdditiveGP=2,
              LearnDimGP=3)
