from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Mapping
from decimal import Decimal, InvalidOperation
from pathlib import PurePosixPath
from urllib.parse import unquote, urlparse


def collapse_whitespace(value: object) -> str:
    return " ".join(str(value or "").split())


def casefolded_text(value: object) -> str:
    return collapse_whitespace(
        unicodedata.normalize("NFKC", str(value or "")).casefold()
    )


def normalized_text(
    value: object,
    *,
    preserve: frozenset[str] = frozenset({"+", "#"}),
) -> str:
    source: str = unicodedata.normalize("NFKC", str(value or "")).casefold()
    characters: list[str] = [
        " "
        if unicodedata.category(character).startswith("P") and character not in preserve
        else character
        for character in source
    ]
    return collapse_whitespace("".join(characters))


def normalized_tokens(
    value: object,
    *,
    preserve: frozenset[str] = frozenset({"+", "#"}),
) -> tuple[str, ...]:
    text: str = normalized_text(value, preserve=preserve)
    return tuple(text.split()) if text else ()


def contains_token_sequence(text: object, phrase: object) -> bool:
    text_tokens: tuple[str, ...] = normalized_tokens(text)
    phrase_tokens: tuple[str, ...] = normalized_tokens(phrase)
    if not text_tokens or not phrase_tokens or len(phrase_tokens) > len(text_tokens):
        return False
    width: int = len(phrase_tokens)
    return any(
        text_tokens[index : index + width] == phrase_tokens
        for index in range(len(text_tokens) - width + 1)
    )


def split_on_tokens(
    value: object,
    separators: Iterable[str],
) -> list[str]:
    separator_tokens: set[str] = {
        normalized
        for separator in separators
        if (normalized := normalized_text(separator))
    }
    parts: list[list[str]] = [[]]
    for token in collapse_whitespace(value).split():
        if token.casefold() in separator_tokens:
            if parts[-1]:
                parts.append([])
            continue
        parts[-1].append(token)
    return [" ".join(part).strip() for part in parts if part]


def rewrite_token_sequences(
    tokens: Iterable[str],
    aliases: Mapping[tuple[str, ...], str],
) -> tuple[str, ...]:
    source: tuple[str, ...] = tuple(tokens)
    widths: tuple[int, ...] = tuple(
        sorted({len(key) for key in aliases if key}, reverse=True)
    )
    output: list[str] = []
    index: int = 0
    while index < len(source):
        replacement: str | None = None
        consumed: int = 0
        for width in widths:
            candidate: tuple[str, ...] = source[index : index + width]
            if candidate in aliases:
                replacement = aliases[candidate]
                consumed = width
                break
        if replacement is None:
            output.append(source[index])
            index += 1
            continue
        output.append(replacement)
        index += consumed
    return tuple(output)


def url_path_value_after(url: str | None, marker: str) -> str | None:
    if not url:
        return None
    parts: tuple[str, ...] = tuple(
        part
        for part in PurePosixPath(unquote(urlparse(url).path)).parts
        if part not in {"/", ""}
    )
    marker_key: str = marker.casefold()
    for index, part in enumerate(parts[:-1]):
        if part.casefold() == marker_key:
            value: str = parts[index + 1].strip()
            return value or None
    return None


def count_numeric_measures(
    value: object,
    *,
    units: Iterable[str],
    suffixes: frozenset[str] = frozenset({"%", "x", "k", "m"}),
    currencies: frozenset[str] = frozenset({"$", "€", "£"}),
) -> int:
    raw_tokens: tuple[str, ...] = tuple(
        unicodedata.normalize("NFKC", str(value or "")).split()
    )
    unit_sequences: tuple[tuple[str, ...], ...] = tuple(
        normalized_tokens(unit) for unit in units
    )
    matches: int = 0
    for index, raw_token in enumerate(raw_tokens):
        token: str = raw_token.strip("()[]{}:;!?")
        separate_currency: bool = token in currencies and index + 1 < len(raw_tokens)
        if separate_currency:
            token = raw_tokens[index + 1].strip("()[]{}:;!?")
        has_currency: bool = separate_currency or (
            bool(token) and token[0] in currencies
        )
        if has_currency:
            token = token[1:] if token and token[0] in currencies else token
        numeric_length: int = 0
        for character in token:
            if character.isdigit() or character in {",", "."}:
                numeric_length += 1
                continue
            break
        numeric: str = token[:numeric_length].replace(",", "")
        if not numeric or not any(character.isdigit() for character in numeric):
            continue
        try:
            Decimal(numeric)
        except InvalidOperation:
            continue
        suffix: str = token[numeric_length:].strip(".,:;!?").casefold()
        if has_currency or suffix in suffixes:
            matches += 1
            continue
        remaining: tuple[str, ...] = tuple(
            token.casefold().strip(".,:;!?") for token in raw_tokens[index + 1 :]
        )
        if any(
            unit_sequence and remaining[: len(unit_sequence)] == unit_sequence
            for unit_sequence in unit_sequences
        ):
            matches += 1
            continue
        if any(
            unit_sequence and suffix and (suffix,) == unit_sequence
            for unit_sequence in unit_sequences
        ):
            matches += 1
    return matches


def contains_numeric_measure(
    value: object,
    *,
    units: Iterable[str],
    suffixes: frozenset[str] = frozenset({"%", "x", "k", "m"}),
    currencies: frozenset[str] = frozenset({"$", "€", "£"}),
) -> bool:
    return (
        count_numeric_measures(
            value,
            units=units,
            suffixes=suffixes,
            currencies=currencies,
        )
        > 0
    )
