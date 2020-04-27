def my_enum(**enums):
    return type('Enum', (), enums)


AcquisitionEnum = my_enum(EI=1,
                          LCB=2,
                          AdditiveLCB=3)
