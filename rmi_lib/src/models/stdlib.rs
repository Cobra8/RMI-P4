// < begin copyright >
// Copyright Ryan Marcus 2020
//
// See root directory of this project for license terms.
//
// < end copyright >


#[derive(Debug, PartialEq, Eq, Hash)]
pub enum StdFunctions {
    ADD,
    MULTIPLY,
    FMA,
    EXP1,
    PHI
}

impl StdFunctions {
    pub fn code(&self, ) -> &'static str {
        match self {
            StdFunctions::ADD => {
"
/* ====================== Addition ====================== */

action floating_add(in double_t first, in double_t second, out overflow128_t result) {
    bool first_bigger = first.exponent == second.exponent ? first.mantissa > second.mantissa : first.exponent > second.exponent;
    uint64_t first_mantissa = ((uint64_t) first.mantissa) | HIDDEN_BIT;
    uint64_t second_mantissa = ((uint64_t) second.mantissa) | HIDDEN_BIT;

    if ((first.exponent == 0 && first.mantissa == 0) || (second.exponent == 0 && second.mantissa == 0)) {
        if (first.exponent == 0 && first.mantissa == 0) {
            result = { second.sign, second.exponent, (bit<128>) second_mantissa };
        } else {
            result = { first.sign, first.exponent, (bit<128>) first_mantissa };
        }
        return;
    }

    exponent_t exponent_difference = first_bigger ? (first.exponent - second.exponent) : (second.exponent - first.exponent);
    uint64_t bigger_mantissa = first_bigger ? first_mantissa : second_mantissa;
    uint64_t smaller_mantissa = first_bigger ? second_mantissa : first_mantissa;
    smaller_mantissa = smaller_mantissa >> ((bit<8>) exponent_difference);

    result.sign = first_bigger ? first.sign : second.sign;
    result.exponent = first_bigger ? first.exponent : second.exponent;
    if (first.sign != second.sign) {
        result.mantissa = (bit<128>) (bigger_mantissa - smaller_mantissa);
    } else {
        result.mantissa = (bit<128>) (bigger_mantissa + smaller_mantissa);
    }
}

control FloatingAdder(in double_t first, in double_t second, out double_t result) {
    FloatingNormalizer() normalizer;

    overflow128_t temp;
    apply {
        floating_add(first, second, temp);
        normalizer.apply(temp);
        result = { temp.sign, temp.exponent, (mantissa_t) temp.mantissa };
    }
}
"
            }
            StdFunctions::MULTIPLY => {
"
/* ====================== Multiplication ====================== */

action floating_multiply(in double_t first, in double_t second, out overflow128_t result) {
    if ((first.exponent == 0 && first.mantissa == 0) || (second.exponent == 0 && second.mantissa == 0)) {
        result = { first.sign ^ second.sign, 0, 0 }; return;
    }

    result.sign = first.sign ^ second.sign;
    result.exponent = (first.exponent - EXPONENT_BIAS) + (second.exponent - EXPONENT_BIAS) + EXPONENT_BIAS;

    bit<128> first_mantissa = ((bit<128>) first.mantissa) | (bit<128>) HIDDEN_BIT;
    bit<128> second_mantissa = ((bit<128>) second.mantissa) | (bit<128>) HIDDEN_BIT;

    result.mantissa = (first_mantissa * second_mantissa) >> 52;
}

control FloatingMultiplier(in double_t first, in double_t second, out double_t result) {
    FloatingNormalizer() normalizer;

    overflow128_t temp;
    apply {
        floating_multiply(first, second, temp);
        normalizer.apply(temp);
        result = { temp.sign, temp.exponent, (mantissa_t) temp.mantissa };
    }
}
"
            }
            StdFunctions::FMA => {
"
/* ====================== Fused Multiply-Add ====================== */

control FloatingFusedMultiplyAdd(in double_t x, in double_t y, in double_t z, out double_t result) {
    FloatingAdder() adder_instance;
    FloatingMultiplier() multiplier_instance;

    apply {
        multiplier_instance.apply(x, y, result);
        adder_instance.apply(result, z, result);
    }
}
"
            }
            StdFunctions::EXP1 => {
"
inline double exp1(double x) { return 0; // TODO }
"
            }
            StdFunctions::PHI => {
"
inline double phi(double x) { return 0; // TODO }
"
            }
        }
    }
}
