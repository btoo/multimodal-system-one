"""Explicit spatial/short-answer contracts, with annotation-blind prompts."""
from __future__ import annotations

import ast
import json
import math
import re

from .backbone_study import prompt_for


def task_prompt(row):
    if row['task'] == 'point':
        return ('Locate the UI element requested below in the screenshot. Return only a JSON object '
                'with keys "x" and "y": the coordinates of a point inside that element, normalized to '
                'the full screenshot. Both numbers must be between 0 and 1. The top-left corner is '
                '(0,0) and the bottom-right corner is (1,1). Do not explain.\nRequested element: ' + row['question'])
    if row['task'] == 'chart':
        return row['question'] + '\nAnswer using only the shortest answer value or phrase. Do not explain.'
    if row['task'] == 'choice': return prompt_for(row)
    raise ValueError('Unknown task')


def safe_input(row):
    return {k: row[k] for k in ('task', 'question', 'media', 'choices') if k in row}


def parse_point(text):
    value = text.strip()
    if value.startswith('```') and value.endswith('```'):
        value = re.sub(r'^```(?:json)?\s*', '', value)[:-3].strip()
    try: point = json.loads(value)
    except (ValueError, TypeError): return None
    if not isinstance(point, dict) or set(point) != {'x', 'y'}: return None
    values = [point['x'], point['y']]
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 1 for x in values): return None
    return values


def chart_score(prediction, answers):
    """ChartQA relaxed numeric accuracy (5% relative tolerance), else exact text."""
    def number(text):
        try:
            text = str(text).strip()
            value = float(text[:-1]) / 100 if text.endswith('%') else float(text)
            return value if math.isfinite(value) else None
        except (ValueError, TypeError): return None
    prediction = prediction.strip()
    guess = number(prediction)
    for answer in answers:
        reference = number(answer)
        if guess is not None and reference is not None:
            if (guess == 0 if reference == 0 else abs(guess-reference)/abs(reference) <= .05): return True
        elif prediction.casefold() == str(answer).strip().casefold(): return True
    return False


def pointer_from_script(text):
    """Parse one source pointer action as data. Never execute source scripts."""
    question, separator, script = text.partition('Output Script:')
    if not separator:
        # One publisher subset omits the marker and puts the literal action
        # immediately after its one-line Task. AST checks below still apply.
        question, separator, script = text.partition('\n')
    if not separator or not question.strip().startswith('Task:'): raise ValueError('missing_task_or_script')
    try: tree = ast.parse(script.strip())
    except SyntaxError: raise ValueError('invalid_script_syntax') from None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr): raise ValueError('not_single_action')
    call = tree.body[0].value
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute) or not isinstance(call.func.value, ast.Name): raise ValueError('not_pointer_call')
    if call.func.value.id != 'pyautogui' or call.func.attr not in ('click', 'moveTo', 'doubleClick', 'rightClick'):
        raise ValueError('not_pointer_call')
    if len(call.args) != 2 or call.keywords: raise ValueError('unsupported_pointer_arguments')
    if any(not isinstance(x, ast.Constant) or isinstance(x.value, bool) or not isinstance(x.value, (int, float)) or not math.isfinite(x.value) for x in call.args):
        raise ValueError('nonliteral_pointer')
    return question.strip()[len('Task:'):].strip(), [float(x.value) for x in call.args]
