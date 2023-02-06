DEFT English Belief and Sentiment Annotation V2

Catalog No.: LDC2016E27

Linguistic Data Consortium
July 21, 2016

1.0 Introduction

DARPA's Deep Exploration and Filtering of Text (DEFT) program aims to
address remaining capability gaps in state-of-the-art natural language
processing technologies related to inference, causal relationships and
anomaly detection.  In support of DEFT, LDC is providing source data
and core resources for system development.

This package contains belief-sentiment annotation for English produced
for the DARPA DEFT program, specifically to address the needs of the
Belief and Sentiment (BeSt) track of the 2016 TAC KBP
evaluations. This annotation task marked belief and sentiment about
entities, relations, and events, as annotated in the DEFT Rich ERE
annotation. Performing annotation of sentiment and belief where the
target is an ERE-annotated mention facilitates the ability of belief
and sentiment systems to link their output to a knowledge base, as
required by DEFT program goals.

This release uses filenames that preserve the naming conventions used
by DEFT ERE, which varied slightly with different data sets, thus the
filenames in this package are not uniform across documents. They are,
however, uniform in maintaining the transparent connection of
annotation files (ERE and BeSt) to source document IDs.

2.0 Directory Structure

README.txt     - this file

data/
  source/      - directory containing source text files (*.txt, *.xml)
  ere/	       - directory containing ERE annotation files (*.rich_ere.xml)
  annotation/  - directory containing annotation files (*.best.xml)

docs/
  deft_anomaly_belief_sentiment_guidelines_v1.2.pdf  - annotation guidelines
  files.tab - document providing mapping from annotation file names
              to full thread names
  kits.tab - document providing mapping from annotation file names
             to annotation tool kit IDs.

dtd/
  belief_sentiment.2.0.0.dtd - Document Type Definition for
                               annotation xml files
  belief_sentiment.2.0.0.rng - Relax NG schema for
                               annotation xml files
  deft_rich_ere.1.1.dtd - Document Type Definition for
  			  ERE annotation xml files
  df_xml.dtd - Document Type Definition for Discussion Forum
  	       XML files ('ENG_DF_*.xml')

3.0 Data Profile and Format

Genre                   Docs  Tokens  Belief annotations  Sentiment annotations
-------------------------------------------------------------------------------
discussion forum         209  138997               12775                  32605
newswire                  37   17904                4733                   6059

In the above table, "docs" refers to newswire documents, or multipost
documents taken from discussion forum threads. Each newswire document
is in the source directory as a single file. Multipost documents with
hexadecimal names are in the source directory as a single ".cmp.txt"
file, but Multipost documents with names of the form <docid>_<offsets>
correspond to a portion of a discussion forum file in the source directory.
Note that there are fewer files in the source directory than in the ere
or annotation directories, as a number of annotated multiposts are drawn
from the same source thread.

Belief-Sentiment annotation files have a best.xml extension and are in
XML format. For a full description of the structure, elements, and
attributes of the Committed Belief annotation files, please see the
DTD in the docs directory of this release.

3.1 Offset Calculation

All XML annotation files for Belief-Sentiment represent standoff
annotation of source .txt files and use offsets to refer to the text
extents of the annotated proposition head words. The offset gives the
start character and length of the text extent. Adding the length to
the start character offset gives the string end character.  Offset
counting starts from the initial character of the given *.cmp.txt
file, character 0, of the source document and includes newlines, which
are always rendered as a single line-feed character (unix-style).

3.2 Source Text Data

Source documents are in several different formats. All documents use
the same naming scheme as in ERE. Newswire documents are newswire XML.
Additional source documents are Discussion Forum data.  Due to the
length of many discussion forum threads, annotation of entire threads
for DEFT is impractical.  Therefore, LDC has selected units we call
Continuous Multi-Posts (CMPs), which consist of a continuous run of
posts from a single thread.  The length of a CMP is between 100-1000
words.  In the case of a short thread, this may include the entire
thread; in the case of longer threads, the CMP is a truncated version
of the thread (and it is possible that there may be more than one CMP
that comes from a single original thread).  Older CMPs are named with
a hexadecimal string, and can be mapped back to the original full thread
using the files.tab document in the docs directory of this package.
Newer CMPs are named <thread-id>_<beg>-<end>, where "beg" and "end"
are offsets for the beginning and end of the document, respectively.
For these documents, the entire source thread is included as DF XML
(as was done in ERE releases).

Note that each older-style CMP is an XML fragment.  Because of the method
used to extract the text from the original discussion forum thread data,
each CMP file contains residual markup tags and/or character entity
references, but is NOT a full XML document (it is not expected to pass
XML validation), and so should be treated as raw text.

3.3 XML Annotation Data

Character offsets and lengths for text extents are calculated based on
treating the corresponding source data file as "raw" text, without
escaping XML metacharacters.  However, the XML format for annotations
includes an "annotation_text" element, which provides the data content
of the text extent associated with the each annotation, and here, XML
escaping is applied to angle-brackets and ampersand characters as
needed, so that when the annotation xml file is read using an XML
parser, the original "annotation_text" content will be returned,
matching what is found at the given offset in the source text file.

For example, the (raw) source text for a given file might contain a
string like "R&amp;D" (because it is a fragment that was extracted
as-is from a larger XML stream).  If the raw string "R&amp;D" were
part of an annotated text extent, it would appear in an
"<annotation_text>" element of the corresponding annotation XML file
as "R&amp;amp;D", but the "length" attribute would be 7 (corresponding
to the original "R&amp;D" string in the source text). 

Therefore, an XML parser is required to correctly ingest the
"annotation_text" elements from the *.cb.xml files in order for
offsets and lengths to be meaningful.

As it turns out, there are no cases of XML "meta-characters" in the
"annotated_text" elements in this release.  But as a general rule,
ALWAYS use an XML parser when handling files with the ".xml" file
extension, and NEVER use an XML parser on files with the ".txt"
extension.


4.0 Annotation Procedure

Belief-Sentiment annotation has two components: belief and
sentiment. Belief annotation marks the belief-holder's commitment to a
belief in the occurrence of an event (event-target), the participation
of an entity in an annotated event (entity-target), and/or the
existence of a relation (relation-target). There are four categories
of belief annotation:

Committed Belief (CB) -- the holder believes the proposition
with certainty

Non-committed Belief (NCB) -- the holder believes the
proposition to be possibly, but not necessarily, true

Reported Belief (ROB) -- the holder reports the belief as
belonging to someone else, without specifying their own belief or lack
of belief in the proposition

Not Applicable (NA) -- the holder expresses some cognitive
attitude other than belief toward the proposition, such as desire,
intention, or obligation.

In addition to the target and belief-type, the holder of the belief is
explicitly indicated (and in the case of reported belief, a chain of
attribution is annotated), and the polarity of the belief is indicated
(positive polarity means belief, at the indicated level of commitment
that the event/relation/enitity-participation did occur, while
negative polarity means belief that it did not occur.

Sentiment is annotated with entities (independent of their role in an
event or relation), relations, and events as targets. Polarity
indicates positive or negative sentiment, and holder (including chain
of attribution where relevant) is indicated as in belief annotation.

The targets and holders of belief and/or sentiment are entity,
relation, and event mentions annotated in DEFT Rich ERE. Beliefs and
sentiments toward other targets are not annotated.

Please see the annotation guidelines included in the docs directory of
this release for additional details.

5.0 Contact Information

If you have any questions about the data in this release, please
contact the following personnel at LDC.

Jennifer Tracey <garjen@ldc.upenn.edu>
                                        -DEFT Anomaly project manager
Michael Arrigo <marrigo@ldc.upenn.edu>
                                        -DEFT Anomaly annotation coordinator
Stephanie Strassel <strassel@ldc.upenn.edu>
                                        -DEFT project PI



-------------------

README Update Log
  Created: Jennifer Tracey, December 22, 2015
  Updated: Brian Gainor, January 13, 2016
  Updated: Brian Gainor, July 21, 2016
