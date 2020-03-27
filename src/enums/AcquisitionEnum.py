def my_enum(**enums):
    return type('Enum', (), enums)


AcquisitionEnum = my_enum(LCB=1)
