#![allow(non_snake_case)]
use std::f64;

pub(crate) const ML_POSINF: f64 = f64::INFINITY;
pub(crate) const ML_NEGINF: f64 = f64::NEG_INFINITY;
pub(crate) const ML_NAN: f64 = f64::NAN;

pub(crate) const M_LN2: f64 = f64::consts::LN_2;
pub(crate) const M_LOG10_2: f64 = f64::consts::LN_2 / f64::consts::LN_10;
pub(crate) const M_PI: f64 = f64::consts::PI;
pub(crate) const M_LN_2PI: f64 = 1.8378770664093453; // ln(2*pi)
pub(crate) const M_LN_SQRT_2PI: f64 = 0.91893853320467274178; // ln(sqrt(2*pi))
#[allow(non_upper_case_globals)]
pub(crate) const M_LN_SQRT_PId2: f64 = 0.22579135264472743236; // ln(sqrt(pi/2))

pub(crate) fn r_finite(x: f64) -> bool {
    x.is_finite()
}

pub(crate) fn fmax2(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return x + y;
    }
    if x < y { y } else { x }
}

pub(crate) fn fmin2(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return x + y;
    }
    if x < y { x } else { y }
}

pub(crate) fn r_log1_exp(x: f64) -> f64 {
    if x > -M_LN2 {
        (-x.exp_m1()).ln()
    } else {
        (-x.exp()).ln_1p()
    }
}

pub(crate) fn r_d__0(log_p: bool) -> f64 {
    if log_p { ML_NEGINF } else { 0.0 }
}

pub(crate) fn r_d__1(log_p: bool) -> f64 {
    if log_p { 0.0 } else { 1.0 }
}

pub(crate) fn r_d_half(log_p: bool) -> f64 {
    if log_p { -M_LN2 } else { 0.5 }
}

pub(crate) fn r_d_lval(p: f64, lower_tail: bool) -> f64 {
    if lower_tail { p } else { 0.5 - p + 0.5 }
}

pub(crate) fn r_d_cval(p: f64, lower_tail: bool) -> f64 {
    if lower_tail { 0.5 - p + 0.5 } else { p }
}

#[allow(dead_code)]
pub(crate) fn r_d_val(x: f64, log_p: bool) -> f64 {
    if log_p { x.ln() } else { x }
}

#[allow(dead_code)]
pub(crate) fn r_d_qiv(p: f64, log_p: bool) -> f64 {
    if log_p { p.exp() } else { p }
}

#[allow(dead_code)]
pub(crate) fn r_d_exp(x: f64, log_p: bool) -> f64 {
    if log_p { x } else { x.exp() }
}

pub(crate) fn r_d_log(p: f64, log_p: bool) -> f64 {
    if log_p { p } else { p.ln() }
}

pub(crate) fn r_d_lexp(x: f64, log_p: bool) -> f64 {
    if log_p { r_log1_exp(x) } else { (-x).ln_1p() }
}

pub(crate) fn r_dt_0(lower_tail: bool, log_p: bool) -> f64 {
    if lower_tail { r_d__0(log_p) } else { r_d__1(log_p) }
}

pub(crate) fn r_dt_1(lower_tail: bool, log_p: bool) -> f64 {
    if lower_tail { r_d__1(log_p) } else { r_d__0(log_p) }
}

pub(crate) fn r_dt_qiv(p: f64, lower_tail: bool, log_p: bool) -> f64 {
    if log_p {
        if lower_tail { p.exp() } else { -p.exp_m1() }
    } else {
        r_d_lval(p, lower_tail)
    }
}

pub(crate) fn r_dt_civ(p: f64, lower_tail: bool, log_p: bool) -> f64 {
    if log_p {
        if lower_tail { -p.exp_m1() } else { p.exp() }
    } else {
        r_d_cval(p, lower_tail)
    }
}

pub(crate) fn r_dt_log(p: f64, lower_tail: bool, log_p: bool) -> f64 {
    if lower_tail { r_d_log(p, log_p) } else { r_d_lexp(p, log_p) }
}

pub(crate) fn r_dt_clog(p: f64, lower_tail: bool, log_p: bool) -> f64 {
    if lower_tail { r_d_lexp(p, log_p) } else { r_d_log(p, log_p) }
}

pub(crate) fn r_pow_di(mut x: f64, mut n: i32) -> f64 {
    if x.is_nan() {
        return x;
    }
    if n == 0 {
        return 1.0;
    }
    if !r_finite(x) {
        return x.powi(n);
    }
    if n < 0 {
        n = -n;
        x = 1.0 / x;
    }
    let mut pow = 1.0;
    let mut nn = n as u32;
    while nn > 0 {
        if nn & 1 == 1 {
            pow *= x;
        }
        nn >>= 1;
        if nn == 0 {
            break;
        }
        x *= x;
    }
    pow
}

pub(crate) fn d1mach(i: i32) -> f64 {
    match i {
        1 => f64::MIN_POSITIVE,
        2 => f64::MAX,
        3 => 0.5 * f64::EPSILON,
        4 => f64::EPSILON,
        5 => M_LOG10_2,
        _ => 0.0,
    }
}

pub(crate) fn i1mach(i: i32) -> i32 {
    match i {
        1 => 5,
        2 => 6,
        3 => 0,
        4 => 0,
        5 => std::mem::size_of::<i32>() as i32 * 8,
        6 => std::mem::size_of::<i32>() as i32,
        7 => 2,
        8 => (std::mem::size_of::<i32>() as i32 * 8) - 1,
        9 => i32::MAX,
        10 => f64::RADIX as i32,
        11 => f32::MANTISSA_DIGITS as i32,
        12 => f32::MIN_EXP as i32,
        13 => f32::MAX_EXP as i32,
        14 => f64::MANTISSA_DIGITS as i32,
        15 => f64::MIN_EXP as i32,
        16 => f64::MAX_EXP as i32,
        _ => 0,
    }
}

pub(crate) fn ml_warn_return_nan() -> f64 {
    ML_NAN
}

pub(crate) fn ml_warning(_code: i32, _msg: &str) {
    // Intentionally no-op: caller can decide how to surface warnings.
}

pub(crate) fn sinpi(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if !r_finite(x) {
        return ml_warn_return_nan();
    }
    let mut y = x % 2.0;
    if y <= -1.0 {
        y += 2.0;
    } else if y > 1.0 {
        y -= 2.0;
    }
    if y == 0.0 || y == 1.0 {
        return 0.0;
    }
    if y == 0.5 {
        return 1.0;
    }
    if y == -0.5 {
        return -1.0;
    }
    (M_PI * y).sin()
}

#[allow(dead_code)]
pub(crate) fn cospi(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if !r_finite(x) {
        return ml_warn_return_nan();
    }
    let y = (x.abs()) % 2.0;
    if (y % 1.0) == 0.5 {
        return 0.0;
    }
    if y == 1.0 {
        return -1.0;
    }
    if y == 0.0 {
        return 1.0;
    }
    (M_PI * y).cos()
}

#[allow(dead_code)]
pub(crate) fn chebyshev_init(dos: &[f64], eta: f64) -> i32 {
    if dos.is_empty() {
        return 0;
    }
    let mut err = 0.0;
    let mut i = 0;
    for ii in 1..=dos.len() {
        i = dos.len() - ii;
        err += dos[i].abs();
        if err > eta {
            return i as i32;
        }
    }
    i as i32
}

pub(crate) fn chebyshev_eval(x: f64, a: &[f64], n: i32) -> f64 {
    if n < 1 || n as usize > 1000 {
        return ml_warn_return_nan();
    }
    if x < -1.1 || x > 1.1 {
        return ml_warn_return_nan();
    }
    let twox = x * 2.0;
    let mut b2 = 0.0;
    let mut b1 = 0.0;
    let mut b0 = 0.0;
    let n = n as usize;
    for i in 1..=n {
        b2 = b1;
        b1 = b0;
        b0 = twox * b1 - b2 + a[n - i];
    }
    (b0 - b2) * 0.5
}

#[allow(dead_code)]
pub(crate) fn gammalims() -> (f64, f64) {
    // IEEE_754 defaults from R.
    (-170.5674972726612, 171.61447887182298)
}

pub(crate) fn lgammacor(x: f64) -> f64 {
    const ALGMCS: [f64; 15] = [
        0.1666389480451863247205729650822,
        -0.00001384948176067563840732986059135,
        0.00000009810825646924729426157171547487,
        -0.00000000001809129475572494194263306266719,
        0.000000000000006221098041892605227126015543416,
        -0.0000000000000003399615005417721944303330599666,
        0.000000000000000002683181998482698748957538846666,
        -0.00000000000000000002868042435334643284144622399999,
        0.0000000000000000000003962837061046434803679306666666,
        -0.000000000000000000000006831888753985766870111999999999,
        0.0000000000000000000000001429227355942498147573333333333,
        -0.000000000000000000000000003547598158101070547199999999999,
        0.0000000000000000000000000001025680058010470912,
        -0.0000000000000000000000000000034011022543167488,
        0.0000000000000000000000000000001276642195630062933,
    ];
    const NALGM: i32 = 5;
    const XBIG: f64 = 94906265.62425156;

    if x < 10.0 {
        return ml_warn_return_nan();
    }
    if x < XBIG {
        let tmp = 10.0 / x;
        return chebyshev_eval(tmp * tmp * 2.0 - 1.0, &ALGMCS, NALGM) / x;
    }
    1.0 / (x * 12.0)
}

pub(crate) fn stirlerr(n: f64) -> f64 {
    const S0: f64 = 0.08333333333333333;
    const S1: f64 = 0.002777777777777778;
    const S2: f64 = 0.0007936507936507937;
    const S3: f64 = 0.0005952380952380953;
    const S4: f64 = 0.0008417508417508417;
    const S5: f64 = 0.001917526917526918;
    const S6: f64 = 0.00641025641025641;
    const S7: f64 = 0.029550653594771242;
    const S8: f64 = 0.17964437236883057;
    const S9: f64 = 1.3924322169059011;
    const S10: f64 = 13.402864044168392;
    const S11: f64 = 156.84828462600202;
    const S12: f64 = 2193.1033333333335;
    const S13: f64 = 36108.77125372499;
    const S14: f64 = 691472.2688513131;
    const S15: f64 = 15238221.539407417;
    const S16: f64 = 382900751.39141417;

    const SFERR_HALVES: [f64; 31] = [
        0.0,
        0.1534264097200273452913848,
        0.0810614667953272582196702,
        0.0548141210519176538961390,
        0.0413406959554092940938221,
        0.03316287351993628748511048,
        0.02767792568499833914878929,
        0.02374616365629749597132920,
        0.02079067210376509311152277,
        0.01848845053267318523077934,
        0.01664469118982119216319487,
        0.01513497322191737887351255,
        0.01387612882307074799874573,
        0.01281046524292022692424986,
        0.01189670994589177009505572,
        0.01110455975820691732662991,
        0.010411265261972096497478567,
        0.009799416126158803298389475,
        0.009255462182712732917728637,
        0.008768700134139385462952823,
        0.008330563433362871256469318,
        0.0079341145643140205472481,
        0.007573675487951840794972024,
        0.007244554301320383179543912,
        0.006942840107209529865664152,
        0.006665247032707682442354394,
        0.006408994188004207068439631,
        0.006171712263039457647532867,
        0.005951370112758847735624416,
        0.005746216513010115682023589,
        0.00555473355196280137103869,
    ];

    if n <= 23.5 {
        let nn = n + n;
        if n <= 15.0 && nn == (nn as i32) as f64 {
            return SFERR_HALVES[nn as usize];
        }
        if n <= 5.25 {
            if n >= 1.0 {
                let l_n = n.ln();
                return lgamma(n) + n * (1.0 - l_n) + ldexp(l_n - M_LN_2PI, -1);
            }
            return lgamma1p(n) - (n + 0.5) * n.ln() + n - M_LN_SQRT_2PI;
        }
        let nn = n * n;
        if n > 12.8 {
            return (S0 - (S1 - (S2 - (S3 - (S4 - (S5 - S6 / nn) / nn) / nn) / nn) / nn) / nn) / n;
        }
        if n > 12.3 {
            return (S0 - (S1 - (S2 - (S3 - (S4 - (S5 - (S6 - S7 / nn) / nn) / nn) / nn) / nn) / nn) / nn) / n;
        }
        if n > 8.9 {
            return (S0 - (S1 - (S2 - (S3 - (S4 - (S5 - (S6 - (S7 - S8 / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / n;
        }
        if n > 7.3 {
            return (S0 - (S1 - (S2 - (S3 - (S4 - (S5 - (S6 - (S7 - (S8 - (S9 - S10 / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / n;
        }
        if n > 6.6 {
            return (S0 - (S1 - (S2 - (S3 - (S4 - (S5 - (S6 - (S7 - (S8 - (S9 - (S10 - (S11 - S12 / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / n;
        }
        if n > 6.1 {
            return (S0 - (S1 - (S2 - (S3 - (S4 - (S5 - (S6 - (S7 - (S8 - (S9 - (S10 - (S11 - (S12 - (S13 - S14 / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / n;
        }
        return (S0 - (S1 - (S2 - (S3 - (S4 - (S5 - (S6 - (S7 - (S8 - (S9 - (S10 - (S11 - (S12 - (S13 - (S14 - (S15 - S16 / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / n;
    }

    let nn = n * n;
    if n > 15.7e6 {
        return S0 / n;
    }
    if n > 6180.0 {
        return (S0 - S1 / nn) / n;
    }
    if n > 205.0 {
        return (S0 - (S1 - S2 / nn) / nn) / n;
    }
    if n > 86.0 {
        return (S0 - (S1 - (S2 - S3 / nn) / nn) / nn) / n;
    }
    if n > 27.0 {
        return (S0 - (S1 - (S2 - (S3 - S4 / nn) / nn) / nn) / nn) / n;
    }
    (S0 - (S1 - (S2 - (S3 - (S4 - S5 / nn) / nn) / nn) / nn) / nn) / n
}

fn logcf(x: f64, i: f64, d: f64, eps: f64) -> f64 {
    let mut c1 = 2.0 * d;
    let mut c2 = i + d;
    let mut c4 = c2 + d;
    let mut a1 = c2;
    let mut b1 = i * (c2 - i * x);
    let mut b2 = d * d * x;
    let mut a2 = c4 * c2 - b2;

    b2 = c4 * b1 - i * b2;

    let scalefactor = 4294967296.0_f64.powi(8);

    while (a2 * b1 - a1 * b2).abs() > (eps * b1 * b2).abs() {
        let c3 = c2 * c2 * x;
        c2 += d;
        c4 += d;
        a1 = c4 * a2 - c3 * a1;
        b1 = c4 * b2 - c3 * b1;

        let c3 = c1 * c1 * x;
        c1 += d;
        c4 += d;
        a2 = c4 * a1 - c3 * a2;
        b2 = c4 * b1 - c3 * b2;

        if b2.abs() > scalefactor {
            a1 /= scalefactor;
            b1 /= scalefactor;
            a2 /= scalefactor;
            b2 /= scalefactor;
        } else if b2.abs() < 1.0 / scalefactor {
            a1 *= scalefactor;
            b1 *= scalefactor;
            a2 *= scalefactor;
            b2 *= scalefactor;
        }
    }

    a2 / b2
}

pub(crate) fn log1pmx(x: f64) -> f64 {
    const MIN_LOG1_VALUE: f64 = -0.79149064;
    if x > 1.0 || x < MIN_LOG1_VALUE {
        return (1.0 + x).ln() - x;
    }
    let r = x / (2.0 + x);
    let y = r * r;
    if x.abs() < 1e-2 {
        return r * ((((2.0 / 9.0 * y + 2.0 / 7.0) * y + 2.0 / 5.0) * y + 2.0 / 3.0) * y - x);
    }
    const TOL_LOGCF: f64 = 1e-14;
    r * (2.0 * y * logcf(y, 3.0, 2.0, TOL_LOGCF) - x)
}

pub(crate) fn lgamma1p(a: f64) -> f64 {
    if a.abs() >= 0.5 {
        return lgammafn(a + 1.0);
    }
    const EULER: f64 = 0.5772156649015328606065120900824024;
    const N: usize = 40;
    const COEFFS: [f64; N] = [
        0.3224670334241132182362075833230126,
        0.06735230105319809513324605383715,
        0.02058080842778454787900092413529198,
        0.007385551028673985266273097291406834,
        0.002890510330741523285752988298486755,
        0.001192753911703260977113935692828109,
        0.0005096695247430424223356548135815582,
        0.0002231547584535793797614188036013401,
        0.0000994575127818085337145958900319017,
        0.00004492623673813314170020750240635786,
        0.00002050721277567069155316650397830591,
        0.000009439488275268395903987425104415055,
        0.000004374866789907487804181793223952411,
        0.000002039215753801366236781900709670839,
        0.0000009551412130407419832857179772951265,
        0.0000004492469198764566043294290331193655,
        0.0000002120718480555466586923135901077628,
        0.0000001004322482396809960872083050053344,
        0.0000000476981016936398056576019341724673,
        0.00000002271109460894316491031998116062124,
        0.00000001083865921489695409107491757968159,
        0.000000005183475041970046655121248647057669,
        0.000000002483674543802478317185008663991718,
        0.00000000119214014058609120744254820277464,
        0.0000000005731367241678862013330194857961011,
        0.0000000002759522885124233145178149692816341,
        0.0000000001330476437424448948149715720858008,
        0.00000000006422964563838100022082448087644648,
        0.00000000003104424774732227276239215783404066,
        0.00000000001502138408075414217093301048780668,
        0.000000000007275974480239079662504549924814047,
        0.000000000003527742476575915083615072228655483,
        0.000000000001711991790559617908601084114443031,
        0.0000000000008315385841420284819798357793954418,
        0.0000000000004042200525289440065536008957032895,
        0.0000000000001966475631096616490411045679010286,
        0.00000000000009573630387838555763782200936508615,
        0.00000000000004664076026428374224576492565974577,
        0.00000000000002273736960065972320633279596737272,
        0.00000000000001109139947083452201658320007192334,
    ];
    const C: f64 = 0.2273736845824652515226821577978691e-12;
    const TOL_LOGCF: f64 = 1e-14;

    let mut lgam = C * logcf(-a / 2.0, (N + 2) as f64, 1.0, TOL_LOGCF);
    for i in (0..N).rev() {
        lgam = COEFFS[i] - a * lgam;
    }
    (a * lgam - EULER) * a - log1pmx(a)
}

pub(crate) fn gammafn(x: f64) -> f64 {
    const GAMCS: [f64; 42] = [
        0.008571195590989331,
        0.004415381324841007,
        0.05685043681599363,
        -0.0042198353964185605,
        0.0013268081812124602,
        -0.00018930245297988804,
        0.00003606925327441245,
        -0.000006056761904460864,
        0.0000010558295463022833,
        -0.0000001811967365542384,
        0.000000031177249647153223,
        -0.000000005354219639019687,
        0.0000000009193275519859589,
        -0.00000000015779412802883398,
        0.000000000027079806229349545,
        -0.00000000000464681865382573,
        0.000000000000797335019200742,
        -0.0000000000001368078209830916,
        0.00000000000002347319486563801,
        -0.000000000000004027432614949067,
        0.0000000000000006910051747372101,
        -0.00000000000000011855845002219929,
        0.00000000000000002034148542496374,
        -0.000000000000000003490054341717406,
        0.0000000000000000005987993856485306,
        -0.00000000000000000010273780578722281,
        0.000000000000000000017627028160605298,
        -0.0000000000000000000030243206537353063,
        0.0000000000000000000005188914660218398,
        -0.00000000000000000000008902770842456577,
        0.000000000000000000000015274740684933426,
        -0.000000000000000000000002620731256187363,
        0.00000000000000000000000044964640478305387,
        -0.00000000000000000000000007714712731336878,
        0.00000000000000000000000001323635453126044,
        -0.0000000000000000000000000022709994129429288,
        0.00000000000000000000000000038964189980039914,
        -0.00000000000000000000000000006685198115125953,
        0.000000000000000000000000000011469986631400244,
        -0.0000000000000000000000000000019679385863451347,
        0.0000000000000000000000000000003376448816585338,
        -0.00000000000000000000000000000005793070335782136,
    ];

    let ngam = 22;
    let xmin = -170.5674972726612;
    let xmax = 171.61447887182298;
    let xsml = 2.2474362225598545e-308;
    let dxrel = 1.4901161193847657e-8;

    if x.is_nan() {
        return x;
    }
    if x == 0.0 || (x < 0.0 && x == x.round()) {
        ml_warning(1, "gammafn");
        return ML_NAN;
    }

    let mut y = x.abs();
    if y <= 10.0 {
        let mut n = x as i32;
        if x < 0.0 {
            n -= 1;
        }
        y = x - n as f64;
        n -= 1;
        let mut value = chebyshev_eval(y * 2.0 - 1.0, &GAMCS, ngam) + 0.9375;
        if n == 0 {
            return value;
        }
        if n < 0 {
            if x < -0.5 && ((x - ((x - 0.5) as i32) as f64) / x).abs() < dxrel {
                ml_warning(8, "gammafn");
            }
            if y < xsml {
                ml_warning(2, "gammafn");
                if x > 0.0 {
                    return ML_POSINF;
                }
                return ML_NEGINF;
            }
            let n = -n;
            for i in 0..n {
                value /= x + i as f64;
            }
            return value;
        }
        for i in 1..=n {
            value *= y + i as f64;
        }
        return value;
    }

    if x > xmax {
        return ML_POSINF;
    }
    if x < xmin {
        return 0.0;
    }

    let value = if y <= 50.0 && y == y.floor() {
        let mut v = 1.0;
        for i in 2..(y as i32) {
            v *= i as f64;
        }
        v
    } else {
        ((y - 0.5) * y.ln() - y + M_LN_SQRT_2PI
            + if (2.0 * y) == (2.0 * y).round() { stirlerr(y) } else { lgammacor(y) })
            .exp()
    };

    if x > 0.0 {
        return value;
    }
    if ((x - ((x - 0.5) as i32) as f64) / x).abs() < dxrel {
        ml_warning(8, "gammafn");
    }
    let sinpiy = sinpi(y);
    if sinpiy == 0.0 {
        ml_warning(2, "gammafn");
        return ML_POSINF;
    }
    -M_PI / (y * sinpiy * value)
}

pub(crate) fn lgammafn_sign(x: f64, sgn: &mut i32) -> f64 {
    let xmax = 2.5327372760800758e305;
    let dxrel = 1.490116119384765625e-8;

    *sgn = 1;
    if x.is_nan() {
        return x;
    }
    if x < 0.0 && (x.abs().floor() % 2.0) == 0.0 {
        *sgn = -1;
    }
    if x <= 0.0 && x == x.trunc() {
        return ML_POSINF;
    }
    let y = x.abs();
    if y < 1e-306 {
        return -y.ln();
    }
    if y <= 10.0 {
        return gammafn(x).abs().ln();
    }
    if y > xmax {
        return ML_POSINF;
    }
    if x > 0.0 {
        if x > 1e17 {
            return x * (x.ln() - 1.0);
        }
        if x > 4934720.0 {
            return M_LN_SQRT_2PI + (x - 0.5) * x.ln() - x;
        }
        return M_LN_SQRT_2PI + (x - 0.5) * x.ln() - x + lgammacor(x);
    }
    let sinpiy = sinpi(y).abs();
    if sinpiy == 0.0 {
        ml_warning(2, "lgamma");
        return ML_POSINF;
    }
    let ans = M_LN_SQRT_PId2 + (x - 0.5) * y.ln() - x - sinpiy.ln() - lgammacor(y);
    if ((x - (x - 0.5).trunc()) * ans / x).abs() < dxrel {
        ml_warning(8, "lgamma");
    }
    ans
}

pub(crate) fn lgammafn(x: f64) -> f64 {
    let mut sgn = 1;
    lgammafn_sign(x, &mut sgn)
}

pub(crate) fn lgamma(x: f64) -> f64 {
    lgammafn(x)
}

pub(crate) fn lbeta(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return a + b;
    }
    let mut p = a;
    let mut q = a;
    if b < p {
        p = b;
    }
    if b > q {
        q = b;
    }
    if p < 0.0 {
        return ml_warn_return_nan();
    }
    if p == 0.0 {
        return ML_POSINF;
    }
    if !r_finite(q) {
        return ML_NEGINF;
    }
    if p >= 10.0 {
        let corr = lgammacor(p) + lgammacor(q) - lgammacor(p + q);
        return -0.5 * q.ln() + M_LN_SQRT_2PI + corr
            + (p - 0.5) * (p / (p + q)).ln()
            + q * (-p / (p + q)).ln_1p();
    }
    if q >= 10.0 {
        let corr = lgammacor(q) - lgammacor(p + q);
        return lgammafn(p) + corr + p - p * (p + q).ln()
            + (q - 0.5) * (-p / (p + q)).ln_1p();
    }
    if p < 1e-306 {
        return lgammafn(p) + (lgammafn(q) - lgammafn(p + q));
    }
    (gammafn(p) * (gammafn(q) / gammafn(p + q))).ln()
}

fn ldexp(x: f64, exp: i32) -> f64 {
    x * 2.0_f64.powi(exp)
}
