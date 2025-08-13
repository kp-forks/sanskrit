# Sanskrit Text Repository

This directory contains standardized versions of Sanskrit texts collected from various web sources. See the readme files in the subdirectories for more details about specific sources.

## File Structure

Each file consists of a header section followed by the main text.

### Header Format
Header elements begin with hash symbols (#), followed by a tag (typically "Text", "Author", "Data entry", "Edition", or "Notes") and terminated with a colon. The actual value appears after the colon.

### Text Markup
In addition to lines of Sanskrit text with standard punctuation, the following markup conventions are used:

- `||` = metatext of any kind
- `||page:55` = page 55 in the original edition
- `||volume:2,page:33` = page 33 in the second volume of the original edition
- `||mula:...` = mula text of a commentary
- `||speaker:...` = speaker attribution (e.g., arjuna uvāca); not consistently marked up.

The texts in the VPC (Vedic Prose Corpus) follow a slightly different annotation convention; see its readme.md for details.