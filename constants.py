RAW_PROMPT = lambda prompt: prompt
A_PROMPT = lambda prompt: "a " + prompt
P_PROMPT = lambda prompt: "picture of a " + prompt
PROMPT_ENGINEERING = {
    "l": {
        "csl": RAW_PROMPT,
        "csl_a": A_PROMPT,
        "csl_p": P_PROMPT,
    },
    "b": {
        "csb": RAW_PROMPT,
        "csb_a": A_PROMPT,
        "csb_p": P_PROMPT,
    },
}