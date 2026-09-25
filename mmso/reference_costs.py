"""Published-rate estimates; not a settled invoice or credit balance."""


def openai_cost(usage, spec):
    total, output = usage['input_tokens'], usage['output_tokens']
    details = usage.get('input_tokens_details', {})
    cached, written = details.get('cached_tokens', 0), details.get('cache_write_tokens', 0)
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in (total, output, cached, written)) or cached + written > total:
        raise ValueError('Invalid token usage accounting')
    rate = spec['input_usd_per_million']
    exact = ((total-cached-written) * rate + cached * rate * .1 + written * rate * 1.25 + output * spec['output_usd_per_million']) / 1e6
    upper = (total * rate + written * rate * .25 + output * spec['output_usd_per_million']) / 1e6
    return {'standard_rate_estimate_usd': exact, 'conservative_upper_usd': upper}
