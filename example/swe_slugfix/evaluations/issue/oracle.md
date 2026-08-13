The implementation only trims the input, lowercases it, and replaces literal
space characters one at a time. It does not treat punctuation and underscores
as separators, does not collapse a run of separators into one hyphen, and can
leave repeated or existing hyphens in the output.
