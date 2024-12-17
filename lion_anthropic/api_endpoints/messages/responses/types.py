from enum import Enum


class StopReason(str, Enum):
    """
    Enumeration of possible reasons why the model stopped generating.

    Attributes:
        END_TURN: the model reached a natural stopping point
        MAX_TOKENS: exceeded the requested max_tokens or the model's maximum
        STOP_SEQUENCE: one of the provided custom stop_sequences was generated
        TOOL_USE: the model invoked one or more tools
    """

    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    TOOL_USE = "tool_use"