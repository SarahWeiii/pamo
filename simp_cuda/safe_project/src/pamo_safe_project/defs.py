class BlockTypes:
    FEM = 0
    ABD_AFFINE = 1
    PG_CONTACT = 2
    PT_CONTACT = 3
    EE_CONTACT = 4
    PP_CONTACT = 5
    PE_CONTACT = 6
    CONSTRAINT = 7


class PTContactTypes:
    PT = 0
    PP01 = 1
    PP02 = 2
    PE012 = 3
    PP03 = 4
    PE031 = 5
    PE023 = 6


class EEContactTypes:
    PP02 = 0
    PP03 = 1
    PP12 = 2
    PP13 = 3
    PE023 = 4
    PE123 = 5
    PE201 = 6
    PE301 = 7
    EE = 8
