import os
import random
import collections
import typing as t

import numpy as np
import transformers
import datasets
import nltk
import tokenizers
import regex
import colorama

import segmentador


VOCAB_SIZE = 6000
MARKER_VALID = "\u2713"
MARKER_NOISE_START = "\u274Cs__"
MARKER_NOISE_END = "\u274Ce__"
MARKER_INTENDED_CORRUPTION = "\u25BC"

SPECIAL_SYMBOLS = {
    MARKER_VALID: 1,
    MARKER_NOISE_START: 2,
    MARKER_NOISE_END: 3,
}


random.seed(7899)
print("Marker symbol (valid):", MARKER_VALID)
print("Marker symbol (noise):", MARKER_NOISE_START, MARKER_NOISE_END)

# Tokenizer available in: https://github.com/ulysses-camara/ulysses-segmenter?tab=readme-ov-file#available-models
seg_model = segmentador.Segmenter(uri_tokenizer=f"tokenizers/{VOCAB_SIZE}_subwords")


DEBUG_PATTERN = "(?:\s*[0-9]{1,3}(?:_[A-Z]{1,30})+\s*)"
ALL_SPECIAL_MARKERS = f"(?:{MARKER_INTENDED_CORRUPTION}|{MARKER_NOISE_START}|{MARKER_NOISE_END}|{MARKER_VALID})"
ALL_BUT_NEWSEG = f"[^{MARKER_VALID}]"
COMMISSION_LIST = r"""
    (?:
    AGRICULTURA(?:[,\s]|DE)*PECU[AÁ]RIA(?:[,\s]|DE)*ABASTECIMENTO[E\s]*(?:DES\s*\.|DESENVOLVIMENTO)\s*RURAL|
    CI[EÊ]NCIA[E\s]*TECNOLOGIA(?:[,\s]|DE)*COMUNICA[CÇ][AÃ]O(?:[E\s]|DA)*INFORM[AÁ]TICA|
    CONSTITUI[CÇ][AÃ]O[E\s]*JUSTI[CÇ]A[E\s]*(?:DE\s*CIDADANIA)?|
    CULTURA|
    DEFESA\s*(?:DO|AO)\s*CONSUMIDOR|
    (?:DES\s*\.|DESENVOLVIMENTO)\s*ECON[OÔ]MICO(?:[,\s]|DE)*IND[UÚ]STRIA(?:[,\s]|DE)*COM[EÉ]RCIO(?:[E\s]|DE)*SERVI[CÇ]OS|
    (?:DES\s*\.|DESENVOLVIMENTO)\s*URBANO|
    DIREITOS\s*DA\s*MULHER|
    DIREITOS\s*DA\s*PESSOA\s*IDOSA|
    DIREITOS\s*DAS\s*PESSOAS\s*COM\s*DEFICI[EÊ]NCIA|
    DIREITOS\s*HUMANOS(?:(?:[E\s]|DAS)*MINORIAS)?|
    EDUCA[CÇ][AÃ]O|
    ESPORTE|
    FINAN[CÇ]AS(?:[E\s]|DE)*TRIBUTA[CÇ][AÃ]O|
    FISCALIZA[CÇ][AÃ]O\s*FINANCEIRA(?:[E\s]|DE)*CONTROLE|
    INTEGRA[CÇ][AÃ]O\s*NACIONAL(?:[,\s]|DE)*(?:DES\s*\.|DESENVOLVIMENTO)\s*REGIONAL(?:[E\s]|DA)*AMAZ[OÔ]NIA|
    LEGISLA[CÇ][AÃ]O\s*PARTICIPATIVA|
    (?:MEIO\s*)?AMBIENTE(?:[E\s]|DE)*DESENVOLVIMENTO\s*SUSTENT[AÁ]VEL|
    MINAS(?:[E\s]|DA)*ENERGIA|
    RELA[CÇ][OÕ]ES\s*EXTERIORES(?:(?:[E\s]|DE)*\s*DEFESA\s*NACIONAL)?|
    SEGURAN[CÇ]A\s*P[UÚ]BLICA[E\s]*COMBATE\s*AO\s*CRIME\s*ORGANIZADO|
    SEGURIDADE\s*SOCIAL(?:[E\s]|DA)*FAM[IÍ]LIA|
    TRABALHO(?:[,\s]|DE)*ADMINISTRA[CÇ][AÃ]O(?:[E\s]|DE)*SERVI[CÇ]O\s*P[UÚ]BLICO|
    TURISMO|
    VIA[CÇ][AÃ]O[E\s]*TRANSPORTES|
    INQU[EÉ]RITO|
    REDA[CÇ][ÃA]O
    )
    """.replace(
    " ", ""
).replace(
    "\n", ""
)

COMMISSIONS = (
    r"COMISS(?:[AÃ]O|[OÕ]ES)[\s:]*" + r"(?:" + r"(?:(?:D[EOA]S?|[\s;:,]|E|PARLAMENTAR(?:ES)?)\s*)+" + COMMISSION_LIST + r"\s*)+"
)


class DetectRecurrentMetadata:

    RE_CAMARA_RAW = regex.compile("C[AÂ]MARA\s*DOS\s*DEPUTADOS")

    RE_BLANK_SPACES = regex.compile(r"\s+")

    @classmethod
    def _detect_camara_recurrent_metadata(cls, subpattern, text, dir_: t.Literal[-1, 1]):
        positions = [match.end() if dir_ == 1 else match.start() for match in cls.RE_CAMARA_RAW.finditer(text)]

        if len(positions) <= 1:
            return text

        ref_pos = max(positions) if dir_ == 1 else min(positions)
        i = 0
        ind_last_space = 0

        while 0 <= i + ref_pos < len(text):
            chrs = {text[j + i].lower() for j in positions}

            if len(chrs) != 1:
                break

            if text[i + ref_pos] == " ":
                ind_last_space = i

            i += 1 * dir_

        if i + ref_pos in {-1, len(text)}:
            chrs = {text[j + i].lower() for j in positions if 0 <= i + j < len(text)}

            if len(chrs) == 1 and chrs.pop() == " ":
                ind_last_space = i

        if dir_ == 1:
            slice_ = text[positions[0] : positions[0] + ind_last_space]

        else:
            slice_ = text[positions[0] + ind_last_space + 1 : positions[0]]

        tokens = [
            f"(\s*{regex.escape(tok)}\s*)"
            for tok in regex.split(r"([^" + UPPERCASE_LETTERS + r"]{1,5})", slice_, flags=regex.IGNORECASE)
            if tok
        ]

        if not tokens:
            return text

        mod_subpattern = subpattern.replace(
            r"\1",
            "".join(
                map(
                    lambda gn: (f"{MARKER_INTENDED_CORRUPTION}\g<{gn}>{MARKER_INTENDED_CORRUPTION}"),
                    range(1, 1 + len(tokens)),
                )
            ),
        )

        text = regex.sub(
            (f"(?<=C[AÂ]MARA\s*DOS\s*DEPUTADOS\s*)" if dir_ == 1 else "")
            + "".join(tokens)
            + (f"(?=\s*C[AÂ]MARA\s*DOS\s*DEPUTADOS)" if dir_ == -1 else ""),
            mod_subpattern,
            text,
        )

        return text

    @classmethod
    def sub(cls, subpattern: str, text: str, *args, **kwargs):
        text = cls._detect_camara_recurrent_metadata(subpattern, text, dir_=1)
        text = cls._detect_camara_recurrent_metadata(subpattern, text, dir_=-1)
        return text


class DetectRecurrentNoise:
    RE_BARCODE = regex.compile(
        r"\*" + f"(?:\s*{MARKER_NOISE_END}\s*{DEBUG_PATTERN}*)?" + r"([\sA-Z0-9]+)" + r"\*",
        regex.IGNORECASE,
    )
    RE_BARCODE_2 = regex.compile(r"(((?:[0-9A-F]{2}\s*?){7})\s*\2)")
    RE_PREAMBLE = regex.compile(
        r"^\s*(" + ALL_BUT_NEWSEG + r"{,60}?)[\s0-9]*" + r"(?=C[aâ]mara\s*dos\s*deputados\s*(Proj|Req))",
        regex.IGNORECASE,
    )
    RE_CAMARA_REPEATED = regex.compile(
        r"(?:"
        + r"(C[AÂ]MARA\s*|(?:PAL[AÁ]CIO\s*DO\s*)?CONGRES)(DOS\s*|SO\s*NAC)"
        + r"(DEPUTADOS|IONAL)"
        + r"([\s0-9]+(?![\s0-9]*[-–\.\)]))?"
        + r"(?!"
        + ALL_BUT_NEWSEG
        + r"{,250}?\s*"
        + r"(?:[dD][eE][cC][rR][eE][tT][aA]|[rR][eE][sS][oO][lL][vV][eE])"
        + ALL_BUT_NEWSEG
        + r"{,40}?\s*:\s*)"
        + r")",
    )
    RE_CAMARA_LOWERCASE = regex.compile(
        f"(?<=^|{MARKER_NOISE_END}\s*{DEBUG_PATTERN}*)"
        + r"(\s*[cC][âa]mara)(\s*[dD]os\s*)([dD]eputados)"
        + r"(?!"
        + ALL_BUT_NEWSEG
        + r"{,250}?\s*"
        + r"(?:[dD][eE][cC][rR][eE][tT][aA]|[rR][eE][sS][oO][lL][vV][eE])"
        + ALL_BUT_NEWSEG
        + r"{,40}?\s*:\s*)"
    )
    RE_COMMISSIONS_REPEATED = regex.compile(r"((?<!\(.{,5}\s*)" + COMMISSIONS + r"(?!\s*.{,5}\)))")
    RE_SALA_DAS_SESSOES_CODE = regex.compile(
        r"(?<=Sala\s*das\s*sess[oõ\u0303ô]+es\s*"
        + ALL_BUT_NEWSEG
        + r"{,150}?)([0-9]{1,5}\s*_\s*(?:"
        + MARKER_NOISE_START
        + r")?\s*[0-9]{1,5})",
        regex.IGNORECASE,
    )

    CAMARA_PAGE_NUMBER_SUFFIX = f"(?=\s*(?:{MARKER_NOISE_START}\s*{DEBUG_PATTERN}*)?\s*C[AÂ]MARA\s*DOS\s*DEPUTADOS)"

    RE_CAMARA_PAGE_NUMBER = regex.compile(r"([0-9]+)" + CAMARA_PAGE_NUMBER_SUFFIX)

    FN_PAGE_NUMBER = lambda page_num: (
        r"(P[aá\s]?g(?:ina)?[\.\s:]*)?"
        + f"(?:{MARKER_NOISE_START}\s*{DEBUG_PATTERN}*)?"
        + f"(\s*0?{page_num}\s*)"
        + f"(?:{MARKER_NOISE_END}\s*{DEBUG_PATTERN}*)?"
        + r"(\s*(?:[\\/-]|de)\s*)"
        + f"(?:{MARKER_NOISE_START}\s*{DEBUG_PATTERN}*)?"
        + r"(\s*[0-9]+\s*)"
        + f"(?:{MARKER_NOISE_END}\s*{DEBUG_PATTERN}*)?"
    )
    RE_PAGE_NUMBER_01 = regex.compile(  # Pág: 1 de 3
        f"^\s*{FN_PAGE_NUMBER(1)}|(P[aá]g(?:ina)?[\.\s:]*){FN_PAGE_NUMBER(1)}",
        regex.IGNORECASE,
    )
    RE_BLANK_SPACES = regex.compile(r"\s+")

    @classmethod
    def _detect_barcode(cls, subpattern, text):
        pseudo_patterns = cls.RE_BARCODE.findall(text)

        if not pseudo_patterns:
            return text

        pseudo_patterns = map(lambda item: r"\s*".join(cls.RE_BLANK_SPACES.sub("", item)), pseudo_patterns)
        pseudo_patterns = set(pseudo_patterns)

        mod_subpattern = subpattern.replace(r"\1", MARKER_INTENDED_CORRUPTION + r"\1" + MARKER_INTENDED_CORRUPTION)

        for barcode in sorted(pseudo_patterns):
            text = regex.sub(f"([\*\s]*{barcode}[\*\s]*)", mod_subpattern, text)

        return text

    @classmethod
    def _detect_preamble_noise(cls, subpattern, text):
        preamble = cls.RE_PREAMBLE.match(text)

        if not preamble or not preamble.group(1).strip():
            return text

        preamble_content = r"\s*".join(preamble.group(1).split(" "))
        preamble_content = regex.escape(preamble_content)
        text = regex.sub(r"(\s*" + preamble_content + r"[\s\d]*)", subpattern, text)
        return text

    @classmethod
    def _detect_camara_page_number(cls, subpattern, text):
        numbers = sorted(map(int, cls.RE_CAMARA_PAGE_NUMBER.findall(text)))

        for i, p in enumerate(numbers, 1):
            if i != p:
                break

            text = regex.sub(f"({i}){cls.CAMARA_PAGE_NUMBER_SUFFIX}", subpattern, text)

        return text

    @classmethod
    def _detect_repeated_camara(cls, subpattern, text):
        mod_subpattern = subpattern.replace(
            r"\1", r"\1" + MARKER_INTENDED_CORRUPTION + r"\2" + MARKER_INTENDED_CORRUPTION + r"\3\4"
        )

        text, sub_count = cls.RE_CAMARA_REPEATED.subn(mod_subpattern, text)

        return text, sub_count

    @classmethod
    def _detect_repeated_camara_lowercase(cls, subpattern, text):
        match = cls.RE_CAMARA_LOWERCASE.match(text)

        if match is None:
            return text

        mod_subpattern = subpattern.replace(
            r"\1", r"\1" + MARKER_INTENDED_CORRUPTION + r"\2" + MARKER_INTENDED_CORRUPTION + r"\3"
        )

        text = cls.RE_CAMARA_LOWERCASE.sub(mod_subpattern, text)

        return text

    @classmethod
    def _detect_repeated_commissions(cls, subpattern, text):
        freqs = collections.Counter(map(str.strip, cls.RE_COMMISSIONS_REPEATED.findall(text)))

        for commission_name, freq in freqs.items():
            if freq <= 2:
                continue

            mod_subpattern = f" {MARKER_INTENDED_CORRUPTION}".join(cls.RE_BLANK_SPACES.split(commission_name))
            mod_subpattern = subpattern.replace(r"\1", mod_subpattern)

            text = text.replace(commission_name, mod_subpattern)

        return text

    @classmethod
    def _detect_page_number(cls, subpattern, text):
        match = cls.RE_PAGE_NUMBER_01.search(text)

        if match is None:
            return text

        last_page = int(match.group(4) or match.group(9))

        mod_subpattern = subpattern.replace(r"\1", r"\1\2\3\4")

        for i in range(1, 1 + last_page):
            text = regex.sub(cls.FN_PAGE_NUMBER(i), mod_subpattern, text, flags=regex.IGNORECASE)

        return text

    @classmethod
    def sub(cls, subpattern: str, text: str, *args, **kwargs):
        text = cls._detect_barcode(subpattern, text)
        text = cls._detect_page_number(subpattern, text)
        text = cls._detect_camara_page_number(subpattern, text)
        text = cls._detect_repeated_camara_lowercase(subpattern, text)
        text = cls._detect_preamble_noise(subpattern, text)
        text = cls.RE_SALA_DAS_SESSOES_CODE.sub(subpattern, text)
        text, _ = cls._detect_repeated_camara(subpattern, text)
        text = cls._detect_repeated_commissions(subpattern, text)
        text = cls.RE_BARCODE_2.sub(subpattern, text)
        return text


class PostProcRecurrentNoise(DetectRecurrentNoise):
    @classmethod
    def subn(cls, subpattern: str, text: str, *args, **kwargs):
        text, sub_count = cls._detect_repeated_camara(subpattern, text)
        return text, sub_count


UPPERCASE_LETTERS = "ÀÁÂÃÇÉÊẼÓÕÔÜÚÍA-Z\u0303\u0300\u0301\u0302\u0303\u0304\u0305\u0340\u0341\u0342\u0343"
UPPERCASE_LETTERS_OR_NUM = UPPERCASE_LETTERS + r"0-9"
VALID_ROMAN_NUM = r"(?:M{0,3}(?:C[MD]|D?C{0,3})(?:X[CL]|L?X{0,3})(?:I?X|I?V|V?I{1,3}))"
NRO_SMALL = r"[nN]\s*[oO0º°\.]{1,3}"
NRO = (
    r"(?:"
    + r"(?:(?<=\s)[dD][eE]\s+)?"
    + r"[nN](?:[uú]me)?(?:ro)?[\.\s]*[oO0º°]"
    + r"(?=[^"
    + MARKER_VALID
    + UPPERCASE_LETTERS
    + UPPERCASE_LETTERS.lower()
    + r"])|"
    + r"(?<=\s)"
    + NRO_SMALL
    + r")"
)
QUOTES = r"”“”\"'‘"
QUOTES_CLASS = f"[{QUOTES}]"


class AgreementList:
    ITEMS = (
        r"(\s*(?:"
        + r"(?:[ÓO]rg[aã]o\s*)?(?:Superior|[cC]oncedente|[cC]onve(?:nente|niada))|"
        + NRO
        + r"\s*(?:SIAFI|Original|Do\s*conv[eê]nio)|"
        + r"Valor\s*(?:do\s*conv[eê]nio)?|"
        + r"(?:In[ií]cio|Fim)\s*(?:d[ea]\s*vig[eê]ncia)?|"
        + r"Objeto|"
        + r"Conv[eê]nio|"
        + r"Processo|"
        + r"Total\s*de\s*itens\s*Licitados|"
        + r"Fundamento\s*legal|"
        + r"Contratada|"
        + r"Questionamentos|"
        + r"Justificativa"
        r")\s*)"
    )
    RE_ITEMS = regex.compile(ITEMS + r"(\s*:)", regex.IGNORECASE)
    REG_GET_LIST = regex.compile(
        r"("
        + r"(?:"
        + ITEMS
        + r":\s*[^:"
        + MARKER_VALID
        + "]{,150}?){2,10}"
        + ITEMS
        + r":\s*[^:"
        + MARKER_VALID
        + r"]{,150}"
        + r")",
        regex.IGNORECASE,
    )

    @classmethod
    def sub(cls, subpattern, text: str, *args, **kwargs):
        re_match = cls.REG_GET_LIST.search(text)
        if re_match is None:
            return text
        s_start, s_end = re_match.span()
        text_slice = text[s_start:s_end]
        subpattern = subpattern.replace(r"\1", r"\1" + f" {MARKER_INTENDED_CORRUPTION} " + r"\2")
        text_slice = cls.RE_ITEMS.sub(
            subpattern,
            text_slice,
        )
        text = f"{text[:s_start]}{text_slice}{text[s_end:]}"
        return text


STATES_ACRONYM = r"""
(?:
AC|
AL|
AP|
AM|
BA|
CE|
DF|
ES|
GO|
MA|
MT|
MS|
MG|
PA|
PB|
PR|
PE|
PI|
RJ|
RN|
RS|
RO|
RR|
SC|
SP|
SE|
TO
)
""".replace(
    "\n", ""
).replace(
    " ", ""
)

DOC_ABBVR_LIST = (
    "COM",
    "DCR",
    "DEN",
    "DTQ",
    "DVS",
    "DVT",
    "EMC",
    "EMD",
    "EML",
    "LDO",
    "EMO",
    "EMP",
    "EMR",
    "ERD",
    "ESB",
    "EXP",
    "INA",
    "INC",
    "MPV",
    "MSC",
    "PAR",
    "PDC",
    "PEC",
    "PET",
    "PFC",
    "PLP",
    "PLV",
    "PRC",
    "PRF",
    "PRN",
    "PRO",
    "RCP",
    "REC",
    "REL",
    "REM",
    "REP",
    "REQ",
    "RIC",
    "RPR",
    "SBE",
    "SBT",
    "SDL",
    "LDO",
    "SIT",
    "TCU",
    "SOA",
    "STF",
    "SUG",
    "SUM",
    "CCJ",
    "TER",
    "TVR",
    "VTS",
    "PL",
    "PDL",
)
DOC_ABBVR = r"(?:" + "|".join(DOC_ABBVR_LIST) + r")"
DOC_ABBVR_WITH_SPACES = r"(?:" + r"|".join(map(lambda item: r"\s*".join(["", *item, ""]), DOC_ABBVR_LIST)) + r")"
MINISTRIES = "|".join(
    (
        "MAPA",
        "MC",
        "MCTI",
        "MCom",
        "MinC",
        "MD",
        "MDR",
        "ME",
        "MEC",
        "MI",
        "MJSP",
        "MMA",
        "MME",
        "MMFDH",
        "MRE",
        "MS",
        "MTP",
        "MTur",
        "CGU",
        "SeGov",
        "SGPR",
        "CC",
        "GSI",
        "AGU",
        "MAER",
        "MESA",
        "MINTER",
        "MInfra",
        "MPA",
        "MPS",
        "SMPE",
        "SAE",
        "PR",
        "SEPPIR",
        "SNPM",
        "SRI",
        "SNPTA",
        "SAC",
    )
)

RAW_NUMBER_PREFIXES = r"Art(?:igo)?s?\s*\.?\s*|" + NRO_SMALL + r"|\$|p[aáàâã]?g?\s*\.|cep\s*\.|ltda\s*\."

ORDINAL_POSITIONS = "|".join(
    (
        "primeir[oa]",
        "segund[oa]",
        "terceir[oa]",
        "quart[ao]",
        "quint[ao]",
        "sext[ao]",
        "s[eé]tim[ao]",
        "oitav[ao]",
        "non[ao]",
        "d[eé]cim[ao]",
    )
)

BASE_LEGAL_ITEMS = (
    r"§\s*[0-9]+",
    r"Art(?:igo)?s?\s*\.?\s*(?:(?:[-–º°0-9]+|(?<=igos?|\s+)[A-Z]{1,2})|\.{3}|[uú]nico)",
    r"(?:\(\s*|\s+|" + QUOTES_CLASS + r")(?<!\s*p[áaâãà]?g?\s*\.\s*)(?:[A-Za-z]|[0-9]{1,2})\s*\)",
    r"(?:par[áa]grafo|§)\s*[úu]nico",
    r"(?:par[áa]grafo|§)\s*[0-9]{1,2}[\soO0º°]*[-–:]",
    r"(?:sub)?se[çc][ãa]o",
    r"\(?" + f"{VALID_ROMAN_NUM}" + r"\s*(?:[-–\)\.])",
    r"(?<!" + RAW_NUMBER_PREFIXES + r")\(?\s+[0-9]{1,2}[\s0oOº°]*(?:[-–\)]|\.(?![\.0-9]))",
    r"(?<!" + RAW_NUMBER_PREFIXES + r")\s+[0-9]{1,2}\s*(?:\.[0-9]+){1,2}(?![\.0-9]*,)",
    "emenda\s*" + NRO,
    r"(?:par[áa]grafo|§|Artigo)\s*" + f"(?:{ORDINAL_POSITIONS})",
    f"[^{UPPERCASE_LETTERS_OR_NUM}][A-FH-OQ-Z]\s*\.\s*[0-9]+",
)

MONTHS = (
    "(?:"
    + "|".join(
        (
            r"[jJ]an(?:eiro)?",
            r"[fF]ev(?:ereiro)",
            r"[mM]ar(?:[cç]o)",
            r"[aA]br(?:il)?",
            r"[mM]ai(?:o)?",
            r"[jJ]un(?:ho)?",
            r"[jJ]ul(?:ho)?",
            r"[aA]go(?:sto)?",
            r"[sS]et(?:embro)?",
            r"[oO]ut(?:ubro)?",
            r"[nN]ov(?:embro)?",
            r"[dD]ez(?:embro)?",
        )
    ).upper()
    + ")"
)

DATE = (
    r"(?:" + r"\s*(?:em|de)?\s*"
    r"(?:"
    + r"[,\s]*[0-9]{1,2}[-–/\.;][0-9]{1,2}[-–/\.;][0-9]{2,4}|"
    + r"[,\s]*(?:(?:de|em|/)[,\.º/0-9\s]*){1,3}[0-9]{4}|"
    + r"[,\s]*(?:de|em|/)?\s*[0-9]{,2}[º°oO\s]*(?:de|em|/)\s*(?:"
    + MONTHS
    + r")\s*(?:de|em|/)\s*[0-9]{4}"
    + r")"
    + r")"
)

DATE_OR_UNDERSCORES = (
    r"(?:" + r"\s*(?:em|de)?\s*"
    r"(?:"
    + r"[,\s]*[_0-9]{1,2}[-–/\.;][_0-9]{1,2}[-–/\.;][_0-9]{2,4}|"
    + r"[,\s]*(?:(?:de|em|/)[,\.º_/0-9\s]*){1,3}(?:[0-9]{4}|[\._]+)|"
    + r"[,\s]*(?:de|em|/)?\s*(?:[0-9]{,2}|[\._]+)[º°oO\s]*(?:de|em|/)\s*(?:"
    + MONTHS
    + r"|_+)\s*(?:de|em|/)\s*(?:[0-9]{4}|[\._]+)"
    + r")"
    + r")"
)

UPPERCASE_DATE_OR_UNDERSCORES = DATE_OR_UNDERSCORES.replace("em", "EM").replace("de", "DE")
EOF = r".{,450}$"
EOF_OR_DATE = r"(?:" + EOF + r"|" + DATE_OR_UNDERSCORES + r")"
RE_DOC_CODE_PREFIX = (
    r"(?:"
    + r"030|Daniel|[eE]ss|Jaa|ac[fgp]|afpa|cmrv|(da[-–])?conle|[Cc]rps|"
    + r"dennn?er|dpsl?|drb|epo|faa|‘?[Gg]ab|gsl|jaa|jbs|kvp|lgl|mlcl?|"
    + r"mm|pnf|rpb|tksa|[Vv][Pp][Ll][cf]?|wgl"
    + r")"
)
RE_DOC_CODE_CORE = r"(?:pls|mpv|plc|pec|pds|plv|prn|plp|pdl|tema)"
RE_DOC_CODE_SUFFIX = (
    r"(?:(?:" + r"c(?:ompleme?ntar)?|eme(?:nda)?s?|" + r"rev(?:is)?|sub(?:st\.?(?:itutivo)?)?|sust|tt?" + r")\s*?)*"
)
RE_DOC_CODE_FULL = (
    r"("
    + r"(?<=\s)"
    + RE_DOC_CODE_PREFIX
    + "/"
    + RE_DOC_CODE_CORE
    + r"(?:[-–0-9]+)"
    + f"(?:{RE_DOC_CODE_SUFFIX}[-–\s]*?)+"
    + r")"
)
EXTRA_LEGAL_ITEMS = (
    r"•",
    r"●",
    "\uF0B7",
)

# CEP 70.160.900
CEP_NUMBERS = r"(?<g_cep_fst>[0-9]{2}\.?[0-9]{3})(?<g_cep_snd>[-–\.\s]*[0-9]{2,3})"
CEP = r"(?:" + r"(?<g_cep_lab>(?:CEP|C[oó]digo\s*[pP]ostal)[-–\s\.:]*)?" + CEP_NUMBERS + r")"

BRASILIA = (
    r"(?:"
    + r"(?<g_bra_name>Bras[ií]lia"
    + ALL_BUT_NEWSEG
    + r"{,5}?)?"
    + r"(?<g_bra_df>(?<=[^"
    + UPPERCASE_LETTERS
    + MARKER_VALID
    + r"])DF|Distrito\s*Federal)"
    + r")"
)

NOISE_PLACE_ITEMS = (
    r"(?:"
    + f"(?:sala|gabinete)\s+({NRO}\s*)?[{UPPERCASE_LETTERS_OR_NUM}]"
    + r"{1,3}(?:[-–\.]"
    + f"[{UPPERCASE_LETTERS_OR_NUM}]"
    + r")?|"
    + r"pavimento\s*(?:s(?:uperior)?|t[eé]rreo)?|"
    + f"(?:Bloco|(?<=[^{UPPERCASE_LETTERS}])Ala)\s+[A-Z](?=[^{UPPERCASE_LETTERS}])|"
    + f"anexo\s+(?:(\s*{NRO}\s*)?[0-9]+|"
    + VALID_ROMAN_NUM
    + r")"
    + r")"
)
NOISE_PLACE_SEP = r"[^" + MARKER_VALID + MARKER_NOISE_START[0] + MARKER_NOISE_END[0] + "]{,40}"

LARGER_BLOCKS_HIERARCHY = (
    "(?:PARTE\s*(?:PRIMEIRA|SEGUNDA|TERCEIRA|QUARTA|QUINTA)\s*(?:DO\s*)?)?LIVRO",
    "T[IÍ]TULO",
    "CAP[IÍ]TULO",
    "(?:Sub)?[sS]e[cç][aã]o",
    BASE_LEGAL_ITEMS[1] + r"(?=\s*[^" + UPPERCASE_LETTERS_OR_NUM + r"])",
)

SOURCE_URL = (
    r"(?:"
    + r"(?:(?:"
    + r"(?:"
    + r"Dispon[ií]vel|Ler|Leia|mais|Vide|Veja|Fontes?|Extra[ií]do|"
    + r"Link|URL|Endere[cç]o|Eletr[oô]nico|Dados|Matéria|Material|"
    + r"Pesquisa|Ver|Publicado|[ÌI]ntegra|Respostas?|Confira|Conferir"
    + r")"
    + r"(?:[,\s]|em|d?[aeo]s?|n[ao]s?|[ao]s)*)+\s*"
    + ALL_BUT_NEWSEG
    + r"{,60}?[\s:]*)?"
    + r"[\<\s]*"
    + r"(?:https?://|www){1,2}"
    + r"(?:[^\s"
    + MARKER_VALID
    + r"]+|\s+\&(?=[\sa-z]*=)|\s*[a-z]+=[^\&\s"
    + MARKER_VALID
    + r"]"
    + r"{,100}\&|(?<=\&)\s*[a-z]+=)*"
    + r"(?:[,\.\s]*acess(?:ado|o)\s*em[\s:]*"
    + DATE_OR_UNDERSCORES
    + r")?"
    + r")"
)

RE_NOISE_BLOCKS = (
    regex.compile(  # 0, Câmara dos Deputados , Gab . 862 , Anexo IV
        r"((?:C[aâ]mara\s*dos\s*Deputados\s*" + ALL_BUT_NEWSEG + r"{,15}?\s*)?" + r"(?:"
        r"Anexo\s*"
        + VALID_ROMAN_NUM
        + r""
        + ALL_BUT_NEWSEG
        + r"{,30}?"
        + r"Gab(?:inete)?.{,10}?"
        + NRO
        + r"?[0-9]+"
        + r"|"
        + r"Gab(?:inete)?.{,10}?"
        + NRO
        + r"?[0-9]+.{,30}?"
        + r"Anexo\s*"
        + VALID_ROMAN_NUM
        + r")"
        + r")",
        regex.IGNORECASE,
    ),
    regex.compile(f"(?<!{NRO}[_X\s\.0-9]*)" + r"([0-9]{11,})"),  # 1
    regex.compile(r"(_{20,}\s*)+"),  # 2
    regex.compile(  # 3
        r"("
        + r"^(?:\s*[^\s"
        + "".join(m[0] for m in ALL_SPECIAL_MARKERS)
        + UPPERCASE_LETTERS_OR_NUM
        + r"]\s*)+|"
        + r"(?:\s*[^\s\.\)\?"
        + "".join(m[0] for m in ALL_SPECIAL_MARKERS)
        + UPPERCASE_LETTERS_OR_NUM
        + r"]\s*)+(?:\.docx?\s*)?$"
        + r")",
        regex.IGNORECASE,
    ),
    regex.compile(  # 4
        r"((?:(?:E[-–\s]*mails?|Endere[cç]os?\s*eletr[oô]nicos?)[\s:]*)?"
        + r"[-–a-zA-Z0-9\._]{,40}\s*@\s*(?:[a-zA-Z]{1,15}\.?){1,3})",
        regex.IGNORECASE,
    ),
    *[  # 5-13-16-20
        regex.compile(
            r"(?<=[:\?;\." + QUOTES + r"]\s*(?:e|ou)?\s*)([0-9]+)(?=\s*" + legal_item + r")",
            regex.IGNORECASE,
        )
        for legal_item in (*BASE_LEGAL_ITEMS, *EXTRA_LEGAL_ITEMS, *LARGER_BLOCKS_HIERARCHY[:-1])
    ],
    regex.compile(  # 21
        r"((?<=C[AÂ]MARA\s*DOS\s*DEPUTADOS\s*)CPI\s*(?:da\s*Petrobr[áa]s)?\s*[-–]\s*"
        + r"(LEI\s*ROUANET|Relat[oó]rio\s*Final|EXPLORA[CÇ][AÃ]O\s*SEXUAL\s*DE\s*CRIAN[CÇ]AS\s*E\s*ADOLESCENTES))",
        regex.IGNORECASE,
    ),
    regex.compile(  # 22
        r"(Gabinete\s*d[eoa]\s*deputad[oa]\s*[^0-9" + MARKER_VALID + "]{,50}?[-–\\/]\s*" + STATES_ACRONYM + "(?=\s|$))",
        regex.IGNORECASE,
    ),
    regex.compile(  # 23
        r"(c[âa]mara\s*dos\s*deputados\s*.{,10}?\s*pra[çc]a\s*dos\s*tr[êe]s\s*poderes)",
        regex.IGNORECASE,
    ),
    regex.compile(  # 24
        r"(C:(\\[^\." + MARKER_VALID + "]+)*\.[a-z]+)",
        regex.IGNORECASE,
    ),
    regex.compile(  # 25
        r"(" + r"[\[\(\s]*" + r"[0-9]+" + r"[\]\)\s]*" + r"[" + UPPERCASE_LETTERS + r"]{,15}?" + SOURCE_URL + r")",
        regex.IGNORECASE,
    ),
    regex.compile(  # 26
        r"(Infoleg[^a-z]{,6}Autenticador)",
        regex.IGNORECASE,
    ),
    regex.compile(  # 27
        r"(" + r"(?:" + NOISE_PLACE_ITEMS + NOISE_PLACE_SEP + r"){2,4}" + NOISE_PLACE_ITEMS + r")",
        regex.IGNORECASE,
    ),
    regex.compile(  # 28
        r"("
        + r"(?:formatado|r[ée]cuo)\s*:\s*"
        + r"(?:"
        + r"fonte\s*:\s*(?:[\s0-9]+pt|\(padr[aã]o\)\s*arial)"
        + r"(?:\s*,\s*(?:Negrito|It[aá]lico|cor\s*da\s*fonte\s*:\s*autom[aá]tica))*|"
        + r"n[aã]o\s*cabe[cç]alho\s*diferente\s*na\s*primeira\s*p[aá]gina|"
        + r"justificado|"
        + r"cor\s*da\s*fonte\s*:\s*autom[aá]tica|"
        r"corpo\s*padr[aã]o\s*,\s*[aàá]\s*(?:esquerda|direita)|"
        + r"espaçamento\s*entre\s*linhas\s*:\s*(?:duplo|simples)|"
        + r"espa[cç]o\s*depois\s*de\s*:\s*[0-9]+(?:cm|pt|['\"])|"
        + r"(?:[,\s]*"
        + r"(?:Esquerda|Direita|Inferior|Largura|Altura|Superior|Primeira\s*linha|Espa[cç]o\s*depois\s*de)"
        + r"\s*:\s*"
        + r"[\.,0-9]+\s*(?:['\"]|cm|pt)?[,\s]*)+|"
        r")" + r")",
        regex.IGNORECASE,
    ),
    regex.compile(  # 29
        r"(" + r"\s*".join("LexEdit") + r")",
        regex.IGNORECASE,
    ),
)
STANDARD_PREFIXES = r"(?:^|;(?:\s*e|\s*ou)?|[\.:…\?]|[\(\[\{]\s*(?:NR|AC|JW|\.{3,}|…)\s*[\)\]\}]\s*|" + f"[{QUOTES}]|\uF03F)"
PREFIX_EXTENSIONS = (
    r"(?:(?:"
    + f"[\s{MARKER_INTENDED_CORRUPTION}]*"
    + MARKER_NOISE_START
    + r"\s*"
    + DEBUG_PATTERN
    + r"*"
    + r""
    + ALL_BUT_NEWSEG
    + r"{,900}?"
    + MARKER_NOISE_END
    + r"\s*"
    + DEBUG_PATTERN
    + r"*"
    + f"[\s{MARKER_INTENDED_CORRUPTION}]*"
    + r"))"
)
RE_PRE_BLOCKS = tuple(
    regex.compile(f"(?<={STANDARD_PREFIXES}{PREFIX_EXTENSIONS}?)(?=\s*{pattern})", regex.IGNORECASE)
    for pattern in [
        *BASE_LEGAL_ITEMS,
        *EXTRA_LEGAL_ITEMS,
        r"D[eê][-–]se\s*ao\s*Projeto\s*a\s*seguinte\s*reda[cç][aã]o\s*:",
    ]
)
ADDITIONAL_TITLES = (
    r"(?:"
    + r"Ju[ií]z[ea]?s?|M[\.\s]*M[aª]?[\s\.]*|"
    + r"Doutor[ea]?s?|D\.?r[aª]?s?[\s\.]*|"
    + r"Professor[ea]?s?|Prof[aª]?s?[\s\.]*|"
    + r"Advogad[ao]s?|Adv[\s\.]*|"
    + r"Capit[aã](?:o|es)?|Cap[\s\.]*|"
    + r"Pastor[ea]?s?|Pr[aª]?s?[\s\.]*|"
    + r"Sargent[ao]s?|Sarg[\s\.]*|"
    + r"Reitor[ea]?s?"
    + r")*"
)
ABBVR_EXMO = r"Ex\.?m[aªoº]s?\s*\.?"
ABBVR_EX = r"Ex\.?[aªoº]?s?\s*\.\s*[ºªᵉ]?"
ABBVR_SR = r"S\.?r\.?[aªeᵉ]?s?(?:\s*[/\(]\s*[oa]s?\s*\)?)?"
ABBVR_MM = r"M\.?M\.[aªoº]*"
DEPT_EXTENSION_CORE = (
    r"(?:(?:"
    + ABBVR_SR
    + r"|Senhor[ea]?s?)?[\s\.]*(?:Deputad[oa]s?|Dep\s*\.)\s*"
    + ADDITIONAL_TITLES
    + "|"
    + r"(?:"
    + ABBVR_SR
    + r"|Senhor[ea]?s?)[\s\.]*(?:Deputad[oa]s?|Dep\s*\.)?\s*"
    + ADDITIONAL_TITLES
    + "|"
    + r"mesa\s*(?:diretora)?|"
    + r"(?:MENSAGEM|"
    + DOC_ABBVR
    + ")\s*"
    + NRO
    + r"|"
    + r"poder\s*(?:executivo|legislativo|judici[aá]rio)|"
    + r"CPI|"
    + r"Bancada|"
    + r"PROVENIENTE\s*DA\s*(?:MEDIDA\s*PROVIS[OÓ]RIA|MPV)|"
    + COMMISSIONS
    + r")\s*"
)
# DOS/AS SRS/AS
DEPT_EXTENSION_A = (
    r"[^\("
    + MARKER_VALID
    + r"]{,100}\(\s*(?:D[oa]s?(?:\s*[/\(]\s*[oa]s?\s*\)?)?)?\s*"
    + DEPT_EXTENSION_CORE
    + f"(?:[^{QUOTES}{MARKER_VALID}\)]"
    + r"{1,200})?\)"
    + r"(?!\s*[;:,])"
)
DEPT_EXTENSION_B = (
    r""
    + ALL_BUT_NEWSEG
    + r"{,100}?D[oa]s?(?:\s*[/\(]\s*[oa]s?\s*\)?)?\s*"
    + DEPT_EXTENSION_CORE
    + f"(?:[^{QUOTES}{MARKER_VALID}]"
    + r"{1,100}"
    + f"?(?=[{QUOTES}]))?"
)
DEPT_EXTENSION = f"(?:{DEPT_EXTENSION_A}|{DEPT_EXTENSION_B})"
DATE_AND_ID = (
    r"(?:"
    + r"(?:DE\s*)+?[\._X0-9]+|"
    + f"(?:{NRO}"
    + r"[_X\s\.0-9]*)?\s*(?:"
    + UPPERCASE_DATE_OR_UNDERSCORES
    + r")|"
    + NRO
    + r"[_X\s\.0-9]*"
    + r"(?:[^,"
    + MARKER_VALID
    + r"]{,30}?[,\.]+\s*(?:DE\s*)+?[\._X0-9]+)?"
    + r")"
)
# DATE
fn_lambda_single = lambda symb, deb: f" {symb} {deb} " + r"\1" + f" {symb} {deb} "
fn_lambda_double = lambda symb, deb: f" {symb} {deb} " + r"\1" + f" {symb} {deb} " + r"\2" + f" {symb} {deb} "
fn_lambda_triple = (
    lambda symb, deb: f" {symb} {deb} " + r"\1" + f" {symb} {deb} " + r"\2" + f" {symb} {deb} " + r"\3" + f" {symb} {deb} "
)
fn_lambda_quad = (
    lambda symb, deb: f" {symb} {deb} "
    + r"\1"
    + f" {symb} {deb} "
    + r"\2"
    + f" {symb} {deb} "
    + r"\3"
    + f" {symb} {deb} "
    + r"\4"
    + f" {symb} {deb} "
)

REQUEST_PRESIDENT_OR_MINISTRY_PREFIX = (
    r"(?:"
    + r"(?:\s(?:Ao|[AÁÀ])s?)?\s*"
    + r"(?:\s*"
    + r"(?:"
    + r"Excelent[ií]ssim[oa]s?|"
    + ABBVR_EXMO
    + r"|"
    + r"Merit[ií]ssim[oa]s?|"
    + ABBVR_MM
    + r"|"
    + r"Magn[iíì]fic[ao]s?|"
    r"A\s*sua\s*(?:magnific[eê]ncia|excel[eê]ncia)|"
    + r"(?:Vossa|V\s*\.)\s*(?:excel[eê]ncias?|"
    + ABBVR_EX
    + r")|"
    + r"Senhor[ae]?s?|"
    + ABBVR_SR
    + r")"
    + r"\s*)+"
    + r"[\.\s]*(?:Primeir[oa]s?|Vices?|[-–\s])*"
    + r")"
)

REQUEST_PRESIDENT_OR_MINISTRY_CORE = (
    r"(?:"
    + r"Pres(?:id(?:ent[ae])?)?s?|"
    + r"Min(?:istr[oa])?s?|"
    + r"Advogad[ao]s?\s*Geral\s*da\s*Uni[aã]o|"
    + r"Secret[aá]ri[oa]s?|"
    + r"Reitor[ea]?s?"
    + r")"
)
REQUEST_PRESIDENT_OR_MINISTRY_SUFFIX = r"(?:[^,:;\." + MARKER_VALID + r"]{,75}?[,:;\.])"
REQUEST_PRESIDENT_OR_MINISTRY = (
    "(?:"
    + REQUEST_PRESIDENT_OR_MINISTRY_PREFIX
    + f"{REQUEST_PRESIDENT_OR_MINISTRY_CORE}?"
    + REQUEST_PRESIDENT_OR_MINISTRY_SUFFIX
    + ")"
)
REQUEST_PRONOUN_COLON = (
    "(?:"
    + REQUEST_PRESIDENT_OR_MINISTRY_PREFIX
    + f"{REQUEST_PRESIDENT_OR_MINISTRY_CORE}?"
    + r"[^:"
    + MARKER_VALID
    + "r]{,75}?\s*:"
    + ")"
)

REQUEST_PRESIDENT_OR_MINISTRY_AFFIXED = (
    r"(?:"
    + r"(?<=(?<!"
    + f"{ABBVR_EXMO}|{ABBVR_EX}|{ABBVR_SR}|{ABBVR_MM}"
    + ")\s*\."
    + ALL_BUT_NEWSEG
    + r"{,10}?|\)"
    + ALL_BUT_NEWSEG
    + r"{,10}?)"
    + REQUEST_PRESIDENT_OR_MINISTRY
    + "|"
    + r"(?:(?<=\.\s*)(\s+O\s*)?Requeir(?:o|emos)|Solicit(?:o|amos))"
    + r")"
)
PRACA_DTP = r"Pra[çc]a\s*dos\s*tr[eê]s\s*poderes"
PRACA_DTP_NEIGHBORS = (
    r"(?|"
    + r"(Gabinete\s*)?(Bras[ií]lia)|(D)(F)|(C[aâ]mara\s*Dos)(\s*Deputados)|"
    + r"((?:Pal[aá]cio\s*do\s*)?Congresso\s*)(Nacional)|(Gabinete\s*)(Parlamentar)|"
    + r"(Comiss[aã]o\s*de\s*)(Fiscaliza[cç][aã]o\s*Financeira[e\s]*Controle)"
    + r")"
)

RE_SPECIAL = (
    (
        regex.compile(  # 0
            r"((?:REQUERIMENTO|SOLICITA[CÇ][AÃ]O)\s*DE\s*INFORMA[CÇ](?:[OÕ\u0303]ES|[AÃ]O)"
            + ALL_BUT_NEWSEG
            + r"{,15}?"
            + DATE_AND_ID
            + f"(?:{DEPT_EXTENSION})?"
            + r")\s*"
            + r"("
            + ALL_BUT_NEWSEG
            + r"{60,1000}?)"
            + f"(?={REQUEST_PRESIDENT_OR_MINISTRY_AFFIXED})",
            regex.IGNORECASE,
        ),
        fn_lambda_double,
        1,
    ),
    (
        regex.compile(  # 1
            r"((?:REQUERIMENTO|SOLICITA[CÇ][AÃ]O)\s*DE\s*INFORMA[CÇ](?:[OÕ\u0303]ES|[AÃ]O)"
            + ALL_BUT_NEWSEG
            + r"{,15}?"
            + f"(?:{DATE_AND_ID})?"
            + f"(?:{DEPT_EXTENSION})?"
            + r")\s*"
            + r"("
            + ALL_BUT_NEWSEG
            + r"{,1000}?)"
            + f"(?={REQUEST_PRONOUN_COLON})",
            regex.IGNORECASE,
        ),
        fn_lambda_double,
        1,
    ),
    (
        regex.compile(  # 2
            r"((?:REQUERIMENTO|SOLICITA[CÇ][AÃ]O)"
            + ALL_BUT_NEWSEG
            + r"{,25}?"
            + f"(?:{DATE_AND_ID}|{DEPT_EXTENSION})"
            + r"{1,2}"
            + r")\s*"
            + r"("
            + ALL_BUT_NEWSEG
            + r"{60,1000}?)"
            + f"(?={REQUEST_PRESIDENT_OR_MINISTRY_AFFIXED})",
            regex.IGNORECASE,
        ),
        fn_lambda_double,
        1,
    ),
    (
        regex.compile(  # 3
            r"((?:REQUERIMENTO|SOLICITA[CÇ][AÃ]O)"
            + ALL_BUT_NEWSEG
            + r"{,25}?"
            + f"(?:{DATE_AND_ID})?"
            + f"(?:{DEPT_EXTENSION})?"
            + r")\s*"
            + r"("
            + ALL_BUT_NEWSEG
            + r"{,1000}?)"
            + f"(?={REQUEST_PRONOUN_COLON})",
            regex.IGNORECASE,
        ),
        fn_lambda_double,
        1,
    ),
    (
        regex.compile(  # 4
            r"((?:(?:REQUERIMENTO|SOLICITA[CÇ][AÃ]O)\s*DE\s*)?INDICA[CÇ][AÃ]O[^\."
            + MARKER_VALID
            + r"]{,20}?"
            + DATE_AND_ID
            + f"(?:{DEPT_EXTENSION})?"
            + r")\s*"
            + r"("
            + ALL_BUT_NEWSEG
            + r"{60,1000}?)"
            + f"(?={REQUEST_PRESIDENT_OR_MINISTRY_AFFIXED})",
            regex.IGNORECASE,
        ),
        fn_lambda_double,
        1,
    ),
    (
        regex.compile(  # 5
            r"((?:(?:REQUERIMENTO|SOLICITA[CÇ][AÃ]O)\s*DE\s*)?INDICA[CÇ][AÃ]O[^\."
            + MARKER_VALID
            + r"]{,20}?"
            + f"(?:{DATE_AND_ID})?"
            + f"(?:{DEPT_EXTENSION})?"
            + r")\s*"
            + r"("
            + ALL_BUT_NEWSEG
            + r"{,1000}?)"
            + f"(?={REQUEST_PRONOUN_COLON})",
            regex.IGNORECASE,
        ),
        fn_lambda_double,
        1,
    ),
    (
        regex.compile(  # 6
            r"((?:(?:SUBSTITUTIVO\s*AO\s*)?PROJETO\s*DE\s*)?RESOLU[CÇ][AÃ]O"
            + ALL_BUT_NEWSEG
            + r"{,50}?"
            + DATE_AND_ID
            + f"(?:{DEPT_EXTENSION})?"
            + r")\s*"
            + r"("
            + ALL_BUT_NEWSEG
            + r"{,1000}?)((?:A\s*mesa\s*d)?A\s*C[âa]mara)"
            + r"(\s*dos\s*deputados)([^\."
            + MARKER_VALID
            + r"]*?resolve\s*:)",
            regex.IGNORECASE,
        ),
        lambda symb, deb: (
            f" {symb} {deb} "
            + r"\1"
            + f" {symb} {deb} "
            + r"\2"
            + f" {symb} {deb} "
            + MARKER_INTENDED_CORRUPTION
            + r"\3"
            + MARKER_INTENDED_CORRUPTION
            + r"\4"
            + MARKER_INTENDED_CORRUPTION
            + r"\5"
        ),
        1,
    ),
    (
        regex.compile(  # 7
            r"((?:(?:SUBSTITUTIVO\s*AO\s*)?PROJETO\s*DE\s*)?RESOLU[CÇ][AÃ]O"
            + ALL_BUT_NEWSEG
            + r"{,50}?"
            + DATE_AND_ID
            + f"(?:{DEPT_EXTENSION})?"
            + r")\s*"
            + r"("
            + ALL_BUT_NEWSEG
            + r"{,1000}?)((?:A\s*mesa\s*d)?A\s*C[âa]mara)"
            + r"(\s*dos\s*deputados)([^\."
            + MARKER_VALID
            + "]*?resolve\s*:)",
            regex.IGNORECASE,
        ),
        lambda symb, deb: (
            f" {symb} {deb} "
            + r"\1"
            + f" {symb} {deb} "
            + r"\2"
            + f" {symb} {deb} "
            + MARKER_INTENDED_CORRUPTION
            + r"\3"
            + MARKER_INTENDED_CORRUPTION
            + r"\4"
            + MARKER_INTENDED_CORRUPTION
            + r"\5"
        ),
        1,
    ),
    (
        regex.compile(  # 8
            r"(MEDIDA\s*PROVIS[ÓO]RIA"
            + ALL_BUT_NEWSEG
            + r"{,50}?"
            + DATE_AND_ID
            + r")\s*"
            + r"("
            + ALL_BUT_NEWSEG
            + r"{,1200}?)"
            + r"([OA]\s*President[ea]\s*da\s*rep[úu]blica[^:"
            + MARKER_VALID
            + r"]+?com\s*for[cç]a\s*de\s*lei\s*:)",
            regex.IGNORECASE,
        ),
        fn_lambda_triple,
        1,
    ),
    (
        regex.compile(  # 9
            r"\s*".join(
                [
                    r"(",
                    r"(?:",
                    *r"Documento",
                    r"|",
                    *r"Chancela",
                    r")",
                    *r"eletr",
                    r"[oô]",
                    *r"nic",
                    r"[ao]",
                    r".{,400}?",
                    *r"mesa",
                    NRO,
                    r"[\s0-9]+",
                    r"(?:de|/|\\)",
                    "(?:\s*[0-9]\s*){4}",
                    r"\.",
                    r")",
                ]
            ),
            regex.IGNORECASE,
        ),
        lambda symb_start, symb_end, deb: f" {symb_start} {deb} " + r"\1" + f" {symb_end} {deb} ",
        None,
    ),
    (
        regex.compile(  # 10
            r"("
            + r"(?:"
            + DOC_ABBVR_WITH_SPACES
            + "\s*"
            + f"(?:{NRO})*"
            + r"\s*[\d\s]+/[\s\d]+)?+\s*"
            + r"\s*".join(["", *"Apresenta", "[çc]", "[aã]", *"o:", ""])
            + r"\s*(?:[0-9]\s*){2}"
            + r"\s*/\s*"
            + r"\s*(?:[0-9]\s*){2}"
            + r"\s*/\s*"
            + r"\s*(?:[0-9]\s*){4}"
            + r"\s*"
            + r"\s*(?:[0-9]\s*){2}"
            + r"\s*:\s*"
            + r")"
            + f"({MARKER_NOISE_START}\s*{DEBUG_PATTERN}*)?"
            + r"(\s*[0-9]\s*)"
            + f"({MARKER_NOISE_END}\s*{DEBUG_PATTERN}*)?"
            + r"(\s*[0-9]\s*)"
            + r"(?:"
            + r"([-–]*)"
            + r"("
            + r"\s*".join(["", *"Mesa", ""])
            + r")"
            + r")?"
            + r"([\s0-9]+(?=[\s0-9]*(?:[§"
            + UPPERCASE_LETTERS
            + r"]|$)))?",
            regex.IGNORECASE | regex.MULTILINE,
        ),
        lambda symb_start, symb_end, deb: f" {symb_start} {deb} "
        + r"\1"
        + MARKER_INTENDED_CORRUPTION
        + r"\3\5"
        + MARKER_INTENDED_CORRUPTION
        + r"\6"
        + MARKER_INTENDED_CORRUPTION
        + r"\7\8"
        + f" {symb_end} {deb} ",
        None,
    ),
    (
        DetectRecurrentNoise,  # 11
        lambda symb_start, symb_end, deb: f" {symb_start} {deb} " + r"\1" + f" {symb_end} {deb} ",
        None,
    ),
    (
        regex.compile(r"_________PLACEHOLDER_________"),  # 12
        lambda symb, deb: r"\1" + f" {symb} {deb} " + r"\2",
        None,
    ),
    (
        regex.compile(  # 13
            r"((?<!\s[sS]\s*\.\s*[aA]\s*|[lL][tT][dD][aA]\s*)\.)" + r"(\s{,10}[-–])(?=\s*[" + UPPERCASE_LETTERS + "])"
        ),
        lambda symb, deb: r"\1" + f" {symb} {deb} " + r"\2",
        None,
    ),
    (
        regex.compile(r"(?<=,\s*(?:e|ou)\s*)" + f"(?={BASE_LEGAL_ITEMS[2]})"),  # 14
        lambda symb, deb: f" {symb} {deb} ",
        None,
    ),
    (
        regex.compile(  # 15
            r"(EMI\s*" + DATE_AND_ID + r"\s*[0-9][0-9\s]*" + f"(?:(?:{MINISTRIES})/?)+" + r")"
            r"(\s*[^," + MARKER_VALID + r"]{,50}?,\s*(?:" + DATE + r")[\.\s]*)?"
        ),
        fn_lambda_double,
        None,
    ),
    (
        regex.compile(  # 16
            r"("
            + r"(?:TVR|(?:Ato\s*de\s*)?Concess[aã]o(?:e|\s)*Renova[cç][ãa]o(?:de|\s)*Concess[aã]o(?:de|\s)*Emissora(?:de|\s)*Rádio(?:e|de|\s)*Televisão)\s*"
            + f"(?:{DATE_AND_ID}|{DEPT_EXTENSION}|{NRO}\s*[_X\s\.,X0-9]*)+"
            + r")"
            + r"\s*((?:mensagem|msc[\s\.]*)\s*"
            + NRO
            + "[_\.0-9\s]+/\s*[0-9]{4})"
            + r"\s*((?:aviso|av[\s\.]*)\s*"
            + NRO
            + "[_\.0-9\s]+/\s*[0-9]{4}"
            + r"(?:\s*[-–]\s*C\s*\.\s*Civil)?)",
            regex.IGNORECASE,
        ),
        fn_lambda_triple,
        1,
    ),
    (
        regex.compile(  # 17
            r"((?:SUBSTITUTIVO\s*AO\s*)?PROJETO\s*DE)(\s*"
            + r"(?:"
            + r"LEI(?:\s*COMPLEMENTAR\s*|\s*DA\s*C[AÂ]MARA\s*|\s*DE\s*CONVERS[AÃ]O\s*)*|"
            + r"DECRETO\s*LEGISLATIVO|"
            + r"RESOLU[ÇC][AÃ]O|"
            + r"EMENDA\s*CONSTITUICIONAL|"
            + r"EMENDA\s*[AÁÀ]\s*CONSTITUI[CÇ][AÃ]O|"
            + r"MEDIDA\s*PROVIS[OÓ]RIA"
            r")\s*"
            + f"(?:{DATE_AND_ID})?"
            + f"(?!{DEPT_EXTENSION})"
            + r"\s*[\s"
            + UPPERCASE_LETTERS_OR_NUM
            + r"]{,150}?"
            + r"(?=(?:[OA]\s+)?[\."
            + UPPERCASE_LETTERS
            + "][a-z])"
            + r")"
        ),
        lambda symb, deb: (
            f" {symb} {deb} " + MARKER_INTENDED_CORRUPTION + r"\1" + MARKER_INTENDED_CORRUPTION + r"\2" + f" {symb} {deb} "
        ),
        2,
    ),
    (
        regex.compile(  # 18
            r"((?:SUBSTITUTIVO\s*AO\s*)?PROJETO\s*DE)(\s*"
            + r"(?:"
            + r"LEI(?:\s*COMPLEMENTAR\s*|\s*DA\s*C[AÂ]MARA\s*|\s*DE\s*CONVERS[AÃ]O\s*)*|"
            + r"DECRETO\s*LEGISLATIVO|"
            + r"RESOLU[ÇC][AÃ]O|"
            + r"EMENDA\s*CONSTITUICIONAL|"
            + r"EMENDA\s*[AÁÀ]\s*CONSTITUI[CÇ][AÃ]O|"
            + r"MEDIDA\s*PROVIS[OÓ]RIA"
            r")\s*"
            + f"(?i:{DATE_AND_ID}|{DEPT_EXTENSION})"
            + r"{1,2}"
            + r"\s*[\s"
            + UPPERCASE_LETTERS_OR_NUM
            + r"]{,150}?"
            + r"(?=(?:[OA]\s+)?[\."
            + UPPERCASE_LETTERS
            + "][a-z])"
            + r")"
        ),
        lambda symb, deb: (
            f" {symb} {deb} " + MARKER_INTENDED_CORRUPTION + r"\1" + MARKER_INTENDED_CORRUPTION + r"\2" + f" {symb} {deb} "
        ),
        2,
    ),
    (
        regex.compile(r"(?<=[" + UPPERCASE_LETTERS + "]{3,}\s+)([0-9]{1,2}\s*\.\s+[0-9]+)"),  # 19
        lambda symb, deb: f" {symb} {deb} " + r"\1",
        None,
    ),
    (
        regex.compile(  # 20
            r"(?<=\s|^)(\s*(?:(?:Tel(?:efone)?s?|Fones?|Fax(?:es)?)[\.\s:]*|ou|,)\s*)"
            + r"(?:([^0-9a-z\s"
            + MARKER_VALID
            + r"]?)(\s*(?:0xx)?[0-9]{2,}\s*)([^0-9a-z\s"
            + MARKER_VALID
            + r"]?))?"
            + r"(\s*[0-9]{4,}\s*[-–\.\s]?)(\s*[0-9]{4,})"
            + r"((?:\s*/\s*[0-9]{4,}\s*)*)",
            regex.IGNORECASE,
        ),
        lambda symb_start, symb_end, deb: (
            f" {symb_start} {deb} "
            + r"\1\2"
            + MARKER_INTENDED_CORRUPTION
            + r"\3\4"
            + MARKER_INTENDED_CORRUPTION
            + r"\5"
            + MARKER_INTENDED_CORRUPTION
            + r"\6"
            + MARKER_INTENDED_CORRUPTION
            + r"\7"
            + MARKER_INTENDED_CORRUPTION
            + f" {symb_end} {deb} "
        ),
        None,
    ),
    (
        regex.compile(  # 21
            r"(PROPOSTA\s*DE\s*FISCALIZA[CÇ][AÃ]O\s*E\s*CONTROLE[^\."
            + MARKER_VALID
            + r"]{,20}?"
            + f"\s*(?:{DATE_AND_ID})?\s*"
            + f"\s*(?:{DEPT_EXTENSION})\s*"
            + r")\s*"
            + r"("
            + ALL_BUT_NEWSEG
            + r"{60,1000}?)"
            + f"(?={REQUEST_PRESIDENT_OR_MINISTRY_AFFIXED})",
            regex.IGNORECASE,
        ),
        fn_lambda_double,
        1,
    ),
    (
        regex.compile(  # 22
            r"(PROPOSTA\s*DE\s*FISCALIZA[CÇ][AÃ]O\s*E\s*CONTROLE[^\."
            + MARKER_VALID
            + r"]{,20}?"
            + f"\s*(?:{DATE_AND_ID})?\s*"
            + f"\s*(?:{DEPT_EXTENSION})?\s*"
            + r")\s*"
            + r"("
            + ALL_BUT_NEWSEG
            + r"{,1000}?)"
            + f"(?={REQUEST_PRONOUN_COLON})",
            regex.IGNORECASE,
        ),
        fn_lambda_double,
        1,
    ),
    (
        regex.compile(  # 23
            r"(OF[IÍ]CIO\s*"
            + NRO
            + r""
            + ALL_BUT_NEWSEG
            + r"{,110}?\s*)"
            + r"((?:Bras[ií]lia|Senado\s*Federal)?[,\s]*(?:"
            + DATE_OR_UNDERSCORES
            + r")[\.\s]*)"
            + r"("
            + ALL_BUT_NEWSEG
            + r"{,500}?\s*)"
            + r"(Assunto\s*:\s*"
            + ALL_BUT_NEWSEG
            + r"{,400}?)"
            + f"(?={REQUEST_PRESIDENT_OR_MINISTRY_AFFIXED}|{REQUEST_PRONOUN_COLON})",
            regex.IGNORECASE,
        ),
        fn_lambda_quad,
        1,
    ),
    (
        regex.compile(  # 24
            r"(Atenciosamente\s*)," + r"(\s*" + ALL_BUT_NEWSEG + r"{,250}?" + RE_DOC_CODE_FULL + r")",
            regex.IGNORECASE,
        ),
        lambda symb, deb: (
            f" {symb} {deb} " + MARKER_INTENDED_CORRUPTION + r"\1" + MARKER_INTENDED_CORRUPTION + r",\2" + f" {symb} {deb} "
        ),
        None,
    ),
    (
        regex.compile(  # 25
            r"((?:REQUERIMENTO|SOLICITA[CÇ][AÃ]O)\s*DE\s*INFORMA[CÇ](?:[OÕ\u0303]ES|[AÃ]O)"
            + ALL_BUT_NEWSEG
            + r"{,10}?"
            + DATE_AND_ID
            + f"(?:{DEPT_EXTENSION})?"
            + r"\s*)"
            + r"("
            + ALL_BUT_NEWSEG
            + r"{,1000}?)"
            + r"([ÀÁA]\s*sua\s*excel[eê]ncia"
            + ALL_BUT_NEWSEG
            + r"{,100}?)"
            + r"(?=(?:"
            + REQUEST_PRESIDENT_OR_MINISTRY
            + "[,\s]*)?(?:Requeiro|Solicito))",
            regex.IGNORECASE,
        ),
        fn_lambda_triple,
        1,
    ),
    (
        regex.compile(r"(Autora?\s*:\s*" + ALL_BUT_NEWSEG + r"{,200}?)(\s*Relatora?\s*:)", regex.IGNORECASE),  # 26
        lambda symb, deb: f" {symb} {deb} " + r"\1" + f" {symb} {deb} " + r"\2",
        None,
    ),
    (
        regex.compile(  # 27
            r"(?<=(?:Relatora?|Autora?)\s*:" + ALL_BUT_NEWSEG + r"{,200}?\s+)(" + VALID_ROMAN_NUM + r"[-–\s]+RELAT[OÓ]RIO\s+)",
            regex.IGNORECASE,
        ),
        lambda symb, deb: f" {symb} {deb} " + r"\1",
        None,
    ),
    (AgreementList, lambda symb, deb: f" {symb} {deb} " + r"\1", None),  # 28
    (
        regex.compile(  # 29
            r"(?=Reiterando\s*os\s*votos\s*de\s*apre[cç]o\s*e\s*considera[cç][aã]o)",
            regex.IGNORECASE,
        ),
        lambda symb, deb: f" {symb} {deb} ",
        None,
    ),
    (
        regex.compile(  # 30
            r"(?<=\s|^)(\s*(?:(?:Tel(?:efone)?s?|Fones?|Fax(?:es)?)[\.\s:]*)\s*)?"
            + r"(?:([^0-9a-z"
            + MARKER_VALID
            + r"]?)(\s*(?:0xx)?[0-9]{2,}\s*)([^0-9a-z"
            + MARKER_VALID
            + r"]?))?"
            + r"(\s*[0-9]{4,}\s*[-–\.\s]?)(\s*[0-9]{4,})"
            + r"((?:\s*/\s*[0-9]{4}\s*)*)",
            regex.IGNORECASE,
        ),
        lambda symb_start, symb_end, deb: (
            f" {symb_start} {deb} "
            + r"\1\2"
            + MARKER_INTENDED_CORRUPTION
            + r"\3\4"
            + MARKER_INTENDED_CORRUPTION
            + r"\5"
            + MARKER_INTENDED_CORRUPTION
            + r"\6"
            + MARKER_INTENDED_CORRUPTION
            + r"\7"
            + f" {symb_end} {deb} "
        ),
        None,
    ),
    (
        regex.compile(  # 31
            r"("
            + r"(?:DESPACHO\s*:\s*|\(\s*)?"
            + f"\s*[AÃÁÀ]S\s*{COMMISSIONS}\s*"
            + r"\(\s*"
            + r")"
            + r"(ART(?:IGO)?[\s\.]+)"
            + r"([0-9]+.{,60}?\))"
            + r"(.{,20}?\))?"
            + r"(?=.{,150}$)",
            regex.IGNORECASE,
        ),
        lambda symb, deb: (
            f" {symb} {deb} "
            + r"\1"
            + MARKER_INTENDED_CORRUPTION
            + r"\2"
            + MARKER_INTENDED_CORRUPTION
            + r"\3"
            + MARKER_INTENDED_CORRUPTION
            + r"\4"
            + f" {symb} {deb} "
        ),
        None,
    ),
    (
        regex.compile(  # 32
            PRACA_DTP_NEIGHBORS
            + r"(?P<g_PRACA>.{,6}?"
            + f"{PRACA_DTP})|(?P<g_PRACA>{PRACA_DTP}"
            + r".{,6}?)"
            + PRACA_DTP_NEIGHBORS,
            regex.IGNORECASE,
        ),
        lambda symb_start, symb_end, deb: (
            f" {symb_start} {deb} "
            + r"\1"
            + MARKER_INTENDED_CORRUPTION
            + r"\2\3\4 "
            + MARKER_INTENDED_CORRUPTION
            + r"\5"
            + f" {symb_end} {deb} "
        ),
        None,
    ),
    (
        regex.compile(r"(^\s*[0-9][\s0-9]*|(?<!:[\s0-9_]*)(?:[0-9]+_+)?\s*[0-9][\s0-9]*(?:\.docx?\s*)?$)"),  # 33
        lambda symb_start, symb_end, deb: f" {symb_start} {deb} " + r"\1" + f" {symb_end} {deb} ",
        None,
    ),
    (
        regex.compile(f"({CEP}(?<g_cep_sep>[-–\s]*){BRASILIA})", regex.IGNORECASE),  # 34
        lambda symb_start, symb_end, deb: (
            f" {symb_start} {deb} "
            + MARKER_INTENDED_CORRUPTION
            + r"\g<g_cep_fst>"
            + MARKER_INTENDED_CORRUPTION
            + r"\g<g_cep_snd>"
            + MARKER_INTENDED_CORRUPTION
            + r"\g<g_cep_sep>"
            + MARKER_INTENDED_CORRUPTION
            + r"\g<g_bra_name>"
            + MARKER_INTENDED_CORRUPTION
            + r"\g<g_bra_df>"
            + MARKER_INTENDED_CORRUPTION
            + f" {symb_end} {deb} "
        ),
        None,
    ),
    (
        regex.compile(f"{BRASILIA}(?<g_cep_sep>[-–\s]*){CEP}", regex.IGNORECASE),  # 35
        lambda symb_start, symb_end, deb: (
            f" {symb_start} {deb} "
            + MARKER_INTENDED_CORRUPTION
            + r"\g<g_bra_name>"
            + MARKER_INTENDED_CORRUPTION
            + r"\g<g_bra_df>"
            + MARKER_INTENDED_CORRUPTION
            + MARKER_INTENDED_CORRUPTION
            + r"\g<g_cep_sep>"
            + MARKER_INTENDED_CORRUPTION
            + r"\g<g_cep_fst>"
            + MARKER_INTENDED_CORRUPTION
            + r"\g<g_cep_snd>"
            + f" {symb_end} {deb} "
        ),
        None,
    ),
    (
        regex.compile(  # 36
            r"([:;" + QUOTES + r"\?]\s*" + f"{PREFIX_EXTENSIONS}?)" + r"(\s{,10}[-–])" + f"(?!\s*{MARKER_NOISE_START})"
        ),
        lambda symb, deb: r"\1" + f" {symb} {deb} " + r"\2",
        None,
    ),
    ####################
    (
        regex.compile(  # 37
            f"(?<={MARKER_NOISE_END}\s*{DEBUG_PATTERN}*\s*)"
            + r"("
            + r"(?:(?:web.?|home\.?)?(?:Site|page)|S[ií]tio|Endere[cç]o)s?\s*(?:eletr[oô]nicos?)?[\s:]*"
            + r"(?:https?://)?"
            + r"www\.([^\s\."
            + MARKER_VALID
            + r"]+\.){1,5}[^\s"
            + MARKER_VALID
            + r"]+"
            + r"(?:[,\s\.]*acess(?:ado|o)\s*em[\s:]*"
            + DATE_OR_UNDERSCORES
            + r")?"
            + r")",
            regex.IGNORECASE,
        ),
        lambda symb_start, symb_end, deb: f" {symb_start} {deb} " + r"\1" + f" {symb_end} {deb} ",
        None,
    ),
    (
        regex.compile(  # 38
            f"(?<={MARKER_NOISE_END}\s*{DEBUG_PATTERN}*\s*|"
            + r"\(\s*(?:NR|AC|JW|\.{3})\s*\)\s*)"
            + r"([0-9]+)(?=\s*(?:Art|§|Par[aá]grafo|(?:Sub)?se[cç][aã]o))",
            regex.IGNORECASE,
        ),
        lambda symb_start, symb_end, deb: f" {symb_start} {deb} " + r"\1" + f" {symb_end} {deb} ",
        None,
    ),
    (
        regex.compile(  # 39
            f"(?<={MARKER_NOISE_END}\s*{DEBUG_PATTERN}*)"
            + r"(\s*)([^\s"
            + MARKER_VALID
            + UPPERCASE_LETTERS
            + r"])((?:\s|\2)*)(\s*)"
            + f"(?={MARKER_NOISE_START}\s*{DEBUG_PATTERN}*)",
            regex.IGNORECASE,
        ),
        lambda symb_start, symb_end, deb: f" {symb_start} {deb} " + r"\1\2\3\4" + f" {symb_end} {deb} ",
        None,
    ),
)

RE_PRE_POST_BLOCKS = tuple(
    regex.compile(
        f"{pattern}" + f"(\s*{MARKER_NOISE_START}{ALL_BUT_NEWSEG}*{MARKER_NOISE_END}\s*{DEBUG_PATTERN}*)?",
        reg_flags,
    )
    for pattern, reg_flags in [
        (  # 0
            r"(ACORDO\s*DE\s*[-," + UPPERCASE_LETTERS_OR_NUM + r"\s]+)(?=(?:[OA]\s+)?[" + UPPERCASE_LETTERS + r"][a-z])",
            0,
        ),
        (r"(?<!\(" + ALL_BUT_NEWSEG + r"{,50}?)(" + COMMISSIONS + ")", 0),  # 1
        (  # 2
            r"(\sO\s*Congresso\s*Nacional\s*"
            + ALL_BUT_NEWSEG
            + r"{,250}?\s*"
            + r"\s*(?:decreta|promulga)\s*"
            + ALL_BUT_NEWSEG
            + r"{,40}?\s*:)",
            regex.IGNORECASE,
        ),
        (  # 3
            r"(\sA\s*C[aâ]mara\s*dos\s+deputados\s*"
            + ALL_BUT_NEWSEG
            + r"{,250}?\s*"
            + r"\s*(?:decreta|promulga)\s*"
            + ALL_BUT_NEWSEG
            + r"{,40}?\s*:)",
            regex.IGNORECASE,
        ),
        (  # 4
            r"((?:SUBSTITUTIVO\s*AO\s*)?"
            + r"Projeto\s*de\s*Lei\s*"
            + r"(?:\s*COMPLEMENTAR\s*|\s*DA\s*C[AÂ]MARA\s*|\s*DE\s*CONVERS[AÃ]O\s*)*\s*"
            + f"(?:{DATE_AND_ID})?"
            + r"\s*"
            + DEPT_EXTENSION
            + r")",
            regex.IGNORECASE,
        ),
        (  # 5
            r"((?:SUBSTITUTIVO\s*AO\s*)?Projeto\s*de\s*Decreto\s*Legislativo\s*"
            + DATE_AND_ID
            + f"(?:{DEPT_EXTENSION})?"
            + r")",
            regex.IGNORECASE,
        ),
        (  # 5
            r"((?:SUBSTITUTIVO\s*AO\s*)?Projeto\s*de\s*Resolu[cç][aã]o\s*" + f"(?:{DEPT_EXTENSION}|{DATE_AND_ID})" + r")",
            regex.IGNORECASE,
        ),
        (  # 6
            r"(?<=^[^\(]{,500}?)(Mensagem\s*" + DATE_AND_ID + r"\s*[0-9][0-9\s]*)",
            regex.IGNORECASE,
        ),
        (  # 7
            r"((?:SUBSTITUTIV[AO]\s*[ÁÀA]\s*)?"
            + r"Proposta\s*de\s*emenda\s*(?:cons?titucional|[aàá]\s*constitui[çc][ãa]o).*?"
            + f"(?:{DEPT_EXTENSION})"
            + r")",
            regex.IGNORECASE,
        ),
        *[  # 8, 9, 10
            (
                r"("
                + f"{LARGER_BLOCKS_HIERARCHY[i]}"
                + r"\s*"
                + f"(?:{VALID_ROMAN_NUM}|[0-9]+)"
                + r"(?:[-–\.\s,"
                + UPPERCASE_LETTERS_OR_NUM
                + r"])+?"
                + r"(?:\s*"
                + MARKER_NOISE_START
                + r""
                + ALL_BUT_NEWSEG
                + r"{,800}?"
                + MARKER_NOISE_END
                + r"\s*"
                + f"{DEBUG_PATTERN}*"
                + r"\s*)?"
                + f"(?={MARKER_VALID}|"
                + r"|".join(LARGER_BLOCKS_HIERARCHY[i + 1 :])
                + r")"
                + r")",
                regex.IGNORECASE,
            )
            for i in range(len(LARGER_BLOCKS_HIERARCHY) - 1)
        ],
        (  # 11, Esta lei entra em vigor cento e oitenta dias após a data de sua publicação
            r"(Art"
            + ALL_BUT_NEWSEG
            + r"{,10}?Est[áàãa]\s*"
            + r"(?:lei|EC|Emenda\s*(?:Constitucional|[àaá\s]*constitui[cç][aã]o)|resolu[cç][aã]o)\s*"
            + r"entr[ea]\s*em\s*vigor\s*"
            + ALL_BUT_NEWSEG
            + r"{,100}?\s*(?:data\s*de\s*)sua\s*publica[cç][aã]o\s*(?:\.|$))",
            regex.IGNORECASE,
        ),
        (r"(APRECIA[CÇ][AÃ]O\s*:" + ALL_BUT_NEWSEG + r"{,100})$", 0),
    ]
)

RE_POST_PROCESSING_BLOCKS = (
    (
        PostProcRecurrentNoise,  # 0
        lambda symb_start, symb_end, deb: f" {symb_start} {deb} " + r"\1" + f" {symb_end} {deb} ",
    ),
    (
        regex.compile(  # 1
            r"("
            + f"[^{UPPERCASE_LETTERS_OR_NUM}{MARKER_VALID}]"
            + r"[0-9]"
            + r"[\]\)\s]+"
            + ALL_BUT_NEWSEG
            + r"{,120}?"
            + SOURCE_URL
            + r")",
            regex.IGNORECASE | regex.REVERSE,
        ),
        lambda symb_start, symb_end, deb: (f" {symb_start} {deb} " + r"\1" + f" {symb_end} {deb} "),
    ),
    (
        regex.compile(  # 2
            r"(?<="
            r"(?:^\s*(?!.{,20}C[ÂA]MARA).{,20}?\s*|"
            + f"{MARKER_NOISE_END}\s*{DEBUG_PATTERN}*)"
            + r"\s*"
            + r")"
            + r"("
            + f"(?:Gabinete\s*d[oa]|^\s*|(?<={MARKER_NOISE_END}\s*{DEBUG_PATTERN}*))"
            + r"\s*deputad[oa]\s*(?:federal)?\s*"
            + f"{ALL_BUT_NEWSEG}"
            + r"{,200}?"
            + r")"
            + f"(?={MARKER_VALID}|{MARKER_NOISE_START})",
            regex.IGNORECASE,
        ),
        lambda symb_start, symb_end, deb: (f" {symb_start} {deb} " + r"\1" + f" {symb_end} {deb} "),
    ),
    (
        regex.compile(  # 3
            f"({MARKER_VALID}\s*{DEBUG_PATTERN}*)(\s*)"
            + f"(\s+[{UPPERCASE_LETTERS_OR_NUM}]"
            + r"{1,3}\s+)"
            + f"(?=\s*{MARKER_VALID}|\s*$)",
            regex.IGNORECASE,
        ),
        lambda symb_start, symb_end, deb: (f" {symb_start} {deb} " + r"\3\2" + f" {symb_end} {deb} " + r"\1"),
    ),
    (
        regex.compile(  # 4
            f"(?<={MARKER_NOISE_END}\s*{DEBUG_PATTERN}*)"
            + r"("
            + ALL_BUT_NEWSEG
            + r"{,10}?\s*C[AÂ]MARA\s*DOS\s*DEPUTADOS\s*"
            + ALL_BUT_NEWSEG
            + r"{,10}?)"
            + f"(?={MARKER_VALID}|{MARKER_NOISE_START})",
            regex.IGNORECASE,
        ),
        lambda symb_start, symb_end, deb: (f" {symb_start} {deb} " + r"\1" + f" {symb_end} {deb} "),
    ),
    (
        regex.compile(  # 5
            f"(?<="
            + r"^\s*|"
            + r"(?:"
            + f"(?:^|{MARKER_VALID})\s*{DEBUG_PATTERN}*\s*{ALL_BUT_NEWSEG}"
            + r"{30,}?"
            + r"|"
            + f"{MARKER_NOISE_START}\s*{DEBUG_PATTERN}*\s*{ALL_BUT_NEWSEG}"
            + r"{60,}?"
            + f")"
            + f"{MARKER_NOISE_END}\s*{DEBUG_PATTERN}*\s*"
            + r")"
            + r"([^"
            + MARKER_VALID
            + MARKER_NOISE_START[0]
            + MARKER_NOISE_END[0]
            + r"]{1,90})"
            f"(?="
            + r"\s*$|"
            + f"\s*{MARKER_NOISE_START}\s*{DEBUG_PATTERN}*\s*"
            + r"(?:"
            + ALL_BUT_NEWSEG
            + r"{30,}?"
            + f"(?:\s*{DEBUG_PATTERN}*\s*$|{MARKER_VALID})|"
            + ALL_BUT_NEWSEG
            + r"{60,}?"
            + MARKER_NOISE_END
            + r")"
            + r")",
            regex.IGNORECASE,
        ),
        lambda symb_start, symb_end, deb: (f" {symb_start} {deb} " + r"\1" + f" {symb_end} {deb} "),
    ),
    (
        regex.compile(  # 6
            f"({MARKER_VALID}\s*{DEBUG_PATTERN}*)(\s*)"
            + f"(\s+[{UPPERCASE_LETTERS_OR_NUM}]"
            + r"{1,3}\s+)"
            + f"({MARKER_NOISE_START}{ALL_BUT_NEWSEG}*{MARKER_NOISE_END}\s*{DEBUG_PATTERN}*)"
            f"(?=\s*{MARKER_VALID}|\s*$)",
            regex.IGNORECASE,
        ),
        lambda symb_start, symb_end, deb: (f" {symb_start} {deb} " + r"\3\2" + f" {symb_end} {deb} " + r"\4\1"),
    ),
    (
        regex.compile(f"(?<={MARKER_NOISE_END}\s*{DEBUG_PATTERN}*\s*)" + r"([0-9])" + r"(?=\s)(?!\s*[-–\)\.])"),  # 7
        lambda symb_start, symb_end, deb: f" {symb_start} {deb} " + r"\1" + f" {symb_end} {deb} ",
    ),
    (
        regex.compile(r"(?<=[\.;,:…" + QUOTES + "]\s*)" + r"([0-9]+\s*)" + f"(?={MARKER_NOISE_START})"),  # 8
        lambda symb_start, symb_end, deb: f" {symb_start} {deb} " + r"\1" + f" {symb_end} {deb} ",
    ),
)

RE_HIGH_PRIORITY_BLOCKS = (
    (
        DetectRecurrentMetadata,  # 0, Sala das Sessões , em 28 de abril de 2020
        lambda symb_start, symb_end, deb: f" {symb_start} {deb} " + r"\1" + f" {symb_end} {deb} ",
    ),
    *[
        (
            regex.compile(f"(?<={STANDARD_PREFIXES}{PREFIX_EXTENSIONS}?)(?=\s*{pattern})", regex.IGNORECASE),
            lambda symb, deb: f" {symb} {deb} ",
        )
        for pattern in [
            r"Sala\s*d[ea]s?\s*(?:sess|comiss|reuni)(?:[õôo\u0303]+es|[ãa]o)\s*" + EOF_OR_DATE,
            r"Senado\s*Federal\s*,\s*" + EOF_OR_DATE,
            r"C[aâ]mara\s*dos\s*Deputados\s*,\s*" + EOF_OR_DATE,
            r"Bras[ií]lia\s*,\s*(?:" + DATE_OR_UNDERSCORES + r")\s*",
            r"(?:Atenciosamente|Respeitosam?ente)\s*,",
            r"\sAs?\s*mesas?\s*da\s*c[aâ]mara\s*dos\s*deputados\s*[^:" + MARKER_VALID + r"]{,300}?:",
        ]
    ],
)


RE_POST_BLOCKS = tuple(regex.compile(f"(?<={pattern})", regex.IGNORECASE) for pattern in [])
COALESCE_NOISE = regex.compile(f"{MARKER_NOISE_END}\s*{DEBUG_PATTERN}*\s*{MARKER_NOISE_START}\s*{DEBUG_PATTERN}*")


def regex_legal_item_anymatch(text: str, debug: bool = False, coalesce_noise: bool = True) -> str:
    aid = 0

    for i, (reg, fun) in enumerate(RE_HIGH_PRIORITY_BLOCKS, aid):
        debug_text = f"{i}_HIGH_PTY" if debug else ""
        try:
            pat = fun(MARKER_VALID, debug_text)

        except TypeError:
            pat = fun(MARKER_NOISE_START, MARKER_NOISE_END, debug_text)

        text = reg.sub(pat, text, concurrent=False)

    for i, reg in enumerate(RE_NOISE_BLOCKS, aid):
        debug_text = f"{i}_NOISE" if debug else ""
        text = reg.sub(
            f" {MARKER_NOISE_START} {debug_text} " + r"\1" + f" {MARKER_NOISE_END} {debug_text} ",
            text,
            concurrent=False,
        )

    for i, (reg, fun, sub_count) in enumerate(RE_SPECIAL, aid):
        debug_text = f"{i}_SPECIAL" if debug else ""
        try:
            pat = fun(MARKER_VALID, debug_text)

        except TypeError:
            pat = fun(MARKER_NOISE_START, MARKER_NOISE_END, debug_text)

        text = reg.sub(pat, text, concurrent=False, count=sub_count or 0)

    if coalesce_noise:
        text = COALESCE_NOISE.sub("", text)

    for i, reg in enumerate(RE_PRE_BLOCKS, aid):
        debug_text = f"{i}_PRE" if debug else ""
        text = reg.sub(f" {MARKER_VALID} {debug_text} ", text, concurrent=False)

    for i, reg in enumerate(RE_POST_BLOCKS, aid):
        debug_text = f"{i}_POS" if debug else ""
        text = reg.sub(f" {MARKER_VALID} {debug_text} ", text, concurrent=False)

    for i, reg in enumerate(RE_PRE_POST_BLOCKS, aid):
        debug_text = f"{i}_PRE_POS" if debug else ""
        text = reg.sub(
            f" {MARKER_VALID} {debug_text} " + r"\1\2" + f" {MARKER_VALID} {debug_text} ",
            text,
            concurrent=True,
        )

    if coalesce_noise:
        text = COALESCE_NOISE.sub("", text)

    post_sub_changed = False

    for i, (reg, fun) in enumerate(RE_POST_PROCESSING_BLOCKS, aid):
        debug_text = f"{i}_POST_PROC" if debug else ""
        try:
            pat = fun(MARKER_VALID, debug_text)

        except TypeError:
            pat = fun(MARKER_NOISE_START, MARKER_NOISE_END, debug_text)

        text, post_sub_count = reg.subn(pat, text, concurrent=False)
        post_sub_changed = post_sub_changed or bool(post_sub_count > 0)

    if post_sub_changed:
        if coalesce_noise:
            text = COALESCE_NOISE.sub("", text)

        for i, reg in enumerate(RE_PRE_BLOCKS, aid):
            debug_text = f"{i}_LATE_PRE" if debug else ""
            text = reg.sub(f" {MARKER_VALID} {debug_text} ", text, concurrent=False)

    return text


def preprocess_instance(
    item,
    ind: int,
    print_preprocessed: bool = False,
    debug: bool = False,
    coalesce_noise: bool = True,
):
    preprocessed_text = seg_model.preprocess_legal_text(item["text"])
    preprocessed_text = regex_legal_item_anymatch(preprocessed_text, debug=debug, coalesce_noise=coalesce_noise)
    preprocessed_text = preprocessed_text.replace(MARKER_INTENDED_CORRUPTION, "@" if debug else "")
    tokens = nltk.tokenize.word_tokenize(preprocessed_text, language="portuguese")

    if print_preprocessed:
        print(
            colorama.Fore.WHITE,
            colorama.Style.DIM,
            preprocessed_text,
            colorama.Style.RESET_ALL,
            sep="",
        )

    labels = [0] * len(tokens)

    i = 0
    while i < len(tokens) - 1:
        if tokens[i] in SPECIAL_SYMBOLS:
            cur_token = tokens.pop(i)
            cur_label = labels.pop(i)

            if cur_label == SPECIAL_SYMBOLS[MARKER_VALID] and cur_token == MARKER_NOISE_START:
                labels[i] = SPECIAL_SYMBOLS[MARKER_VALID]
                if i + 1 < len(tokens) and tokens[i + 1] != MARKER_NOISE_END:
                    labels[i + 1] = SPECIAL_SYMBOLS[MARKER_NOISE_START]
                continue

            if cur_label == SPECIAL_SYMBOLS[MARKER_VALID] and cur_token == MARKER_NOISE_END:
                labels[i] = SPECIAL_SYMBOLS[MARKER_VALID]
                if i > 0 and labels[i - 1] != SPECIAL_SYMBOLS[MARKER_NOISE_START]:
                    labels[i - 1] = SPECIAL_SYMBOLS[MARKER_NOISE_END]
                continue

            if cur_label == SPECIAL_SYMBOLS[MARKER_NOISE_START] and cur_token == MARKER_VALID:
                labels[i] = SPECIAL_SYMBOLS[MARKER_VALID]
                if i + 1 < len(tokens) and tokens[i + 1] != MARKER_NOISE_END:
                    labels[i + 1] = SPECIAL_SYMBOLS[MARKER_NOISE_END]
                continue

            if cur_label == SPECIAL_SYMBOLS[MARKER_NOISE_END] and cur_token == MARKER_VALID:
                labels[i] = SPECIAL_SYMBOLS[MARKER_VALID]
                if i > 0 and labels[i - 1] != SPECIAL_SYMBOLS[MARKER_NOISE_START]:
                    labels[i - 1] = SPECIAL_SYMBOLS[MARKER_NOISE_END]
                continue

            if cur_label == SPECIAL_SYMBOLS[MARKER_NOISE_START] and cur_token == MARKER_NOISE_END:
                continue

            if cur_label == SPECIAL_SYMBOLS[MARKER_NOISE_END] and cur_token == MARKER_NOISE_START:
                labels[i] = 0
                continue

            labels[i] = SPECIAL_SYMBOLS[cur_token]
            continue

        i += 1

    if labels:
        maybe_erase_pool = []
        noise_on = False

        for i in range(len(labels) - 1):
            if labels[i] == SPECIAL_SYMBOLS[MARKER_NOISE_END] and labels[i + 1] == SPECIAL_SYMBOLS[MARKER_NOISE_START]:
                labels[i] = labels[i + 1] = 0

        for i in range(len(labels)):
            if labels[i] == SPECIAL_SYMBOLS[MARKER_NOISE_START]:
                maybe_erase_pool.clear()
                continue

            if labels[i] == SPECIAL_SYMBOLS[MARKER_NOISE_END]:
                while maybe_erase_pool:
                    ind = maybe_erase_pool.pop()
                    labels[ind] = 0

            if labels[i] > 0:
                maybe_erase_pool.append(i)

        for i in range(len(labels)):
            if labels[i] == SPECIAL_SYMBOLS[MARKER_NOISE_START]:
                if noise_on:
                    labels[i] = 0
                else:
                    noise_on = True

            elif labels[i] == SPECIAL_SYMBOLS[MARKER_NOISE_END]:
                if noise_on:
                    noise_on = False
                else:
                    labels[i] = 0

            elif labels[i] == SPECIAL_SYMBOLS[MARKER_VALID]:
                noise_on = False

        for i in range(len(labels) - 1):
            if labels[i] == SPECIAL_SYMBOLS[MARKER_NOISE_END] and labels[i + 1] == SPECIAL_SYMBOLS[MARKER_VALID]:
                labels[i] = 0

        while tokens and tokens[0] in SPECIAL_SYMBOLS:
            labels.pop(0)
            tokens.pop(0)

        while tokens and tokens[-1] in SPECIAL_SYMBOLS:
            labels.pop()
            tokens.pop()

        if labels[0] == SPECIAL_SYMBOLS[MARKER_VALID]:
            labels[0] = 0

    ret = {
        "labels": labels,
        "tokens": tokens,
        "id": str(ind),
    }

    return ret


def load_raw_data():
    # Data info + download link: https://github.com/ulysses-camara/ulysses-segmenter?tab=readme-ov-file#train-and-evaluation-data
    df = datasets.load_dataset(
        "csv",
        data_files=["data/ulysses_segmenter_raw_data.txt"],
        header=None,
        names=["text"],
    )

    RE_JUSTIFICATIVA = regex.compile(
        r"\s*(?:"
        + r"J\s*U\s*S\s*T\s*I\s*F\s*I\s*C\s*A?\s*T\s*I\s*[CV]\s*A|"
        + r"J\s*u\s*s\s*t\s*i\s*f\s*i\s*c\s*a\s*t\s*i\s*v\s*a\s+(?=["
        + UPPERCASE_LETTERS
        + r"])|"
        + r"J\s*U\s*S\s*T\s*I\s*F\s*I\s*C\s*A\s*[CÇ]\s*[AÂÃÀÁ]\s*O|"
        + r"J\s*u\s*s\s*t\s*i\s*f\s*i\s*c\s*a\s*[cç]\s*[aãâàá]\s*o\s+(?=["
        + UPPERCASE_LETTERS
        + r"])"
        + r")"
    )

    RE_ANEXO = regex.compile(r"\s*A\s*N\s*E\s*X\s*O")

    df = df.filter(lambda item: isinstance(item["text"], str) and 128 <= len(item["text"]) <= 600000)
    df = df.map(lambda item: {"text": RE_JUSTIFICATIVA.split(item["text"])[0]})
    df = df.map(lambda item: {"text": RE_ANEXO.split(item["text"])[0]})

    df = df.map(preprocess_instance, with_indices=True, num_proc=10, remove_columns="text")

    return df


def tokenize_and_align_labels(examples, max_tokens_per_inst: int = 700):
    new_examples = []
    new_labels = []

    for inst_tokens, inst_labels in zip(examples["tokens"], examples["labels"]):
        propagate_noise_start_to_next_slice = False

        for j in range(0, len(inst_tokens), max_tokens_per_inst):
            slice_tokens = inst_tokens[j : j + max_tokens_per_inst]
            slice_labels = inst_labels[j : j + max_tokens_per_inst]

            if propagate_noise_start_to_next_slice:
                if slice_labels[0] == 0:
                    slice_labels[0] = SPECIAL_SYMBOLS[MARKER_NOISE_START]

                propagate_noise_start_to_next_slice = False

            cur_special_tokens = np.flatnonzero(slice_labels)

            if cur_special_tokens.size and cur_special_tokens[-1] == SPECIAL_SYMBOLS[MARKER_NOISE_START]:
                propagate_noise_start_to_next_slice = True

            new_examples.append(slice_tokens)
            new_labels.append(slice_labels)

    # source: https://huggingface.co/docs/transformers/custom_datasets#preprocess
    tokenized_inputs = seg_model.tokenizer(
        new_examples,
        truncation=True,
        max_length=1024,
        is_split_into_words=True,
    )

    labels = []

    for i, label in enumerate(new_labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels

    return tokenized_inputs


def run():
    df = load_raw_data()

    df_tokenized = df["train"].map(
        tokenize_and_align_labels,
        batched=True,
        num_proc=10,
        remove_columns=df["train"].column_names,
    )

    df_tokenized_train_eval_test = df_tokenized.train_test_split(
        test_size=0.2,
        shuffle=True,
        seed=16,
    )

    df_tokenized_test_eval = df_tokenized_train_eval_test["test"].train_test_split(
        test_size=0.5,
        shuffle=False,
        seed=49,
    )

    df_tokenized_split = datasets.DatasetDict(
        {
            "train": df_tokenized_train_eval_test["train"],
            "eval": df_tokenized_test_eval["train"],
            "test": df_tokenized_test_eval["test"],
        }
    )

    df_tokenized_split.save_to_disk(f"data/df_tokenized_split_0_120000_{VOCAB_SIZE}")


if __name__ == "__main__":
    run()
