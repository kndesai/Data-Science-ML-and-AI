{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "96dbe65e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'\\nTest\\n'"
      ]
     },
     "execution_count": 1,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# -*- coding: utf-8 -*-\n",
    "\"\"\"\n",
    "Twokenize -- a tokenizer designed for Twitter text in English and some other European languages.\n",
    "This tokenizer code has gone through a long history:\n",
    "\n",
    "(1) Brendan O'Connor wrote original version in Python, http://github.com/brendano/tweetmotif\n",
    "       TweetMotif: Exploratory Search and Topic Summarization for Twitter.\n",
    "       Brendan O'Connor, Michel Krieger, and David Ahn.\n",
    "       ICWSM-2010 (demo track), http://brenocon.com/oconnor_krieger_ahn.icwsm2010.tweetmotif.pdf\n",
    "(2a) Kevin Gimpel and Daniel Mills modified it for POS tagging for the CMU ARK Twitter POS Tagger\n",
    "(2b) Jason Baldridge and David Snyder ported it to Scala\n",
    "(3) Brendan bugfixed the Scala port and merged with POS-specific changes\n",
    "    for the CMU ARK Twitter POS Tagger  \n",
    "(4) Tobi Owoputi ported it back to Java and added many improvements (2012-06)\n",
    "\n",
    "Current home is http://github.com/brendano/ark-tweet-nlp and http://www.ark.cs.cmu.edu/TweetNLP\n",
    "\n",
    "There have been at least 2 other Java ports, but they are not in the lineage for the code here.\n",
    "\n",
    "Ported to Python by Myle Ott <myleott@gmail.com>.\n",
    "\n",
    "# Modified by Firoj Alam - Jan, 2017\n",
    "#\n",
    "\"\"\"\n",
    "\n",
    "from __future__ import print_function\n",
    "\n",
    "from __future__ import absolute_import\n",
    "from __future__ import unicode_literals\n",
    "import operator\n",
    "import re\n",
    "import six.moves.html_parser\n",
    "from dateutil.parser import _timelex, parser\n",
    "import string\n",
    "import os\n",
    "import six\n",
    "from six.moves import range\n",
    "\n",
    "import sys\n",
    "# reload(sys)\n",
    "# sys.setdefaultencoding('utf8')\n",
    "\n",
    "\n",
    "def regex_or(*items):\n",
    "    return '(?:' + '|'.join(items) + ')'\n",
    "\n",
    "\n",
    "Contractions = re.compile(\"(?i)(\\w+)(n['’′]t|['’′]ve|['’′]ll|['’′]d|['’′]re|['’′]s|['’′]m)$\", re.UNICODE)\n",
    "Whitespace = re.compile(\"[\\s\\u0020\\u00a0\\u1680\\u180e\\u202f\\u205f\\u3000\\u2000-\\u200a]+\", re.UNICODE)\n",
    "\n",
    "punctChars = r\"['\\\"“”‘’.?!…,:;]\"\n",
    "# punctSeq   = punctChars+\"+\"\t#'anthem'. => ' anthem '.\n",
    "punctSeq = r\"['\\\"“”‘’]+|[.?!,…]+|[:;]+\"  # 'anthem'. => ' anthem ' .\n",
    "entity = r\"&(?:amp|lt|gt|quot);\"\n",
    "#  URLs\n",
    "\n",
    "\n",
    "# BTO 2012-06: everyone thinks the daringfireball regex should be better, but they're wrong.\n",
    "# If you actually empirically test it the results are bad.\n",
    "# Please see https://github.com/brendano/ark-tweet-nlp/pull/9\n",
    "\n",
    "urlStart1 = r\"(?:https?://|\\bwww\\.)\"\n",
    "commonTLDs = r\"(?:com|org|edu|gov|net|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|pro|tel|travel|xxx)\"\n",
    "ccTLDs = r\"(?:ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|\" + \\\n",
    "         r\"bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|\" + \\\n",
    "         r\"er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|\" + \\\n",
    "         r\"hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|\" + \\\n",
    "         r\"lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|\" + \\\n",
    "         r\"nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|sk|\" + \\\n",
    "         r\"sl|sm|sn|so|sr|ss|st|su|sv|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|\" + \\\n",
    "         r\"va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|za|zm|zw)\"  # TODO: remove obscure country domains?\n",
    "urlStart2 = r\"\\b(?:[A-Za-z\\d-])+(?:\\.[A-Za-z0-9]+){0,3}\\.\" + regex_or(commonTLDs,\n",
    "                                                                      ccTLDs) + r\"(?:\\.\" + ccTLDs + r\")?(?=\\W|$)\"\n",
    "urlBody = r\"(?:[^\\.\\s<>][^\\s<>]*?)?\"\n",
    "urlExtraCrapBeforeEnd = regex_or(punctChars, entity) + \"+?\"\n",
    "urlEnd = r\"(?:\\.\\.+|[<>]|\\s|$)\"\n",
    "url = regex_or(urlStart1, urlStart2) + urlBody + \"(?=(?:\" + urlExtraCrapBeforeEnd + \")?\" + urlEnd + \")\"\n",
    "\n",
    "# Numeric\n",
    "timeLike = r\"\\d+(?::\\d+){1,2}\"\n",
    "num = r\"\\d+\"\n",
    "numNum = r\"\\d+\\.\\d+\"\n",
    "numberWithCommas = r\"(?:(?<!\\d)\\d{1,3},)+?\\d{3}\" + r\"(?=(?:[^,\\d]|$))\"\n",
    "numComb = \"[\\u0024\\u058f\\u060b\\u09f2\\u09f3\\u09fb\\u0af1\\u0bf9\\u0e3f\\u17db\\ua838\\ufdfc\\ufe69\\uff04\\uffe0\\uffe1\\uffe5\\uffe6\\u00a2-\\u00a5\\u20a0-\\u20b9]?\\\\d+(?:\\\\.\\\\d+)+%?\"\n",
    "\n",
    "# Abbreviations\n",
    "boundaryNotDot = regex_or(\"$\", r\"\\s\", r\"[“\\\"?!,:;]\", entity)\n",
    "aa1 = r\"(?:[A-Za-z]\\.){2,}(?=\" + boundaryNotDot + \")\"\n",
    "aa2 = r\"[^A-Za-z](?:[A-Za-z]\\.){1,}[A-Za-z](?=\" + boundaryNotDot + \")\"\n",
    "standardAbbreviations = r\"\\b(?:[Mm]r|[Mm]rs|[Mm]s|[Dd]r|[Ss]r|[Jj]r|[Rr]ep|[Ss]en|[Ss]t)\\.\"\n",
    "arbitraryAbbrev = regex_or(aa1, aa2, standardAbbreviations)\n",
    "separators = \"(?:--+|―|—|~|–|=)\"\n",
    "decorations = \"(?:[♫♪]+|[★☆]+|[♥❤♡]+|[\\u2639-\\u263b]+|[\\ue001-\\uebbb]+)\"\n",
    "thingsThatSplitWords = r\"[^\\s\\.,?\\\"]\"\n",
    "embeddedApostrophe = thingsThatSplitWords + r\"+['’′]\" + thingsThatSplitWords + \"*\"\n",
    "\n",
    "#  Emoticons\n",
    "# myleott: in Python the (?iu) flags affect the whole expression\n",
    "# normalEyes = \"(?iu)[:=]\" # 8 and x are eyes but cause problems\n",
    "normalEyes = \"[:=]\"  # 8 and x are eyes but cause problems\n",
    "wink = \"[;]\"\n",
    "noseArea = \"(?:|-|[^a-zA-Z0-9 ])\"  # doesn't get :'-(\n",
    "happyMouths = r\"[D\\)\\]\\}]+\"\n",
    "sadMouths = r\"[\\(\\[\\{]+\"\n",
    "tongue = \"[pPd3]+\"\n",
    "otherMouths = r\"(?:[oO]+|[/\\\\]+|[vV]+|[Ss]+|[|]+)\"  # remove forward slash if http://'s aren't cleaned\n",
    "\n",
    "# mouth repetition examples:\n",
    "# @aliciakeys Put it in a love song :-))\n",
    "# @hellocalyclops =))=))=)) Oh well\n",
    "\n",
    "# myleott: try to be as case insensitive as possible, but still not perfect, e.g., o.O fails\n",
    "# bfLeft = u\"(♥|0|o|°|v|\\\\$|t|x|;|\\u0ca0|@|ʘ|•|・|◕|\\\\^|¬|\\\\*)\".encode('utf-8')\n",
    "bfLeft = \"(♥|0|[oO]|°|[vV]|\\\\$|[tT]|[xX]|;|\\u0ca0|@|ʘ|•|・|◕|\\\\^|¬|\\\\*)\"\n",
    "bfCenter = r\"(?:[\\.]|[_-]+)\"\n",
    "bfRight = r\"\\2\"\n",
    "s3 = r\"(?:--['\\\"])\"\n",
    "s4 = r\"(?:<|&lt;|>|&gt;)[\\._-]+(?:<|&lt;|>|&gt;)\"\n",
    "s5 = \"(?:[.][_]+[.])\"\n",
    "# myleott: in Python the (?i) flag affects the whole expression\n",
    "# basicface = \"(?:(?i)\" +bfLeft+bfCenter+bfRight+ \")|\" +s3+ \"|\" +s4+ \"|\" + s5\n",
    "basicface = \"(?:\" + bfLeft + bfCenter + bfRight + \")|\" + s3 + \"|\" + s4 + \"|\" + s5\n",
    "\n",
    "eeLeft = r\"[＼\\\\ƪԄ\\(（<>;ヽ\\-=~\\*]+\"\n",
    "eeRight = \"[\\\\-=\\\\);'\\u0022<>ʃ）/／ノﾉ丿╯σっµ~\\\\*]+\"\n",
    "eeSymbol = r\"[^A-Za-z0-9\\s\\(\\)\\*:=-]\"\n",
    "eastEmote = eeLeft + \"(?:\" + basicface + \"|\" + eeSymbol + \")+\" + eeRight\n",
    "\n",
    "oOEmote = r\"(?:[oO]\" + bfCenter + r\"[oO])\"\n",
    "\n",
    "emoticon = regex_or(\n",
    "    # Standard version  :) :( :] :D :P\n",
    "    \"(?:>|&gt;)?\" + regex_or(normalEyes, wink) + regex_or(noseArea, \"[Oo]\") + regex_or(tongue + r\"(?=\\W|$|RT|rt|Rt)\",\n",
    "                                                                                       otherMouths + r\"(?=\\W|$|RT|rt|Rt)\",\n",
    "                                                                                       sadMouths, happyMouths),\n",
    "\n",
    "    # reversed version (: D:  use positive lookbehind to remove \"(word):\"\n",
    "    # because eyes on the right side is more ambiguous with the standard usage of : ;\n",
    "    regex_or(\"(?<=(?: ))\", \"(?<=(?:^))\") + regex_or(sadMouths, happyMouths, otherMouths) + noseArea + regex_or(\n",
    "        normalEyes, wink) + \"(?:<|&lt;)?\",\n",
    "\n",
    "    # inspired by http://en.wikipedia.org/wiki/User:Scapler/emoticons#East_Asian_style\n",
    "    eastEmote.replace(\"2\", \"1\", 1), basicface,\n",
    "    # iOS 'emoji' characters (some smileys, some symbols) [\\ue001-\\uebbb]\n",
    "    # TODO should try a big precompiled lexicon from Wikipedia, Dan Ramage told me (BTO) he does this\n",
    "\n",
    "    # myleott: o.O and O.o are two of the biggest sources of differences\n",
    "    #          between this and the Java version. One little hack won't hurt...\n",
    "    oOEmote\n",
    ")\n",
    "\n",
    "Hearts = \"(?:<+/?3+)+\"  # the other hearts are in decorations\n",
    "\n",
    "Arrows = regex_or(r\"(?:<*[-―—=]*>+|<+[-―—=]*>*)\", \"[\\u2190-\\u21ff]+\")\n",
    "\n",
    "# BTO 2011-06: restored Hashtag, AtMention protection (dropped in original scala port) because it fixes\n",
    "# \"hello (#hashtag)\" ==> \"hello (#hashtag )\"  WRONG\n",
    "# \"hello (#hashtag)\" ==> \"hello ( #hashtag )\"  RIGHT\n",
    "# \"hello (@person)\" ==> \"hello (@person )\"  WRONG\n",
    "# \"hello (@person)\" ==> \"hello ( @person )\"  RIGHT\n",
    "# ... Some sort of weird interaction with edgepunct I guess, because edgepunct \n",
    "# has poor content-symbol detection.\n",
    "\n",
    "# This also gets #1 #40 which probably aren't hashtags .. but good as tokens.\n",
    "# If you want good hashtag identification, use a different regex.\n",
    "Hashtag = \"#[a-zA-Z0-9_]+\"  # optional: lookbehind for \\b\n",
    "# optional: lookbehind for \\b, max length 15\n",
    "AtMention = \"[@＠][a-zA-Z0-9_]+\"\n",
    "\n",
    "# I was worried this would conflict with at-mentions\n",
    "# but seems ok in sample of 5800: 7 changes all email fixes\n",
    "# http://www.regular-expressions.info/email.html\n",
    "Bound = r\"(?:\\W|^|$)\"\n",
    "Email = regex_or(\"(?<=(?:\\W))\", \"(?<=(?:^))\") + r\"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,4}(?=\" + Bound + \")\"\n",
    "\n",
    "# We will be tokenizing using these regexps as delimiters\n",
    "# Additionally, these things are \"protected\", meaning they shouldn't be further split themselves.\n",
    "Protected = re.compile(\n",
    "    six.text_type(regex_or(\n",
    "        Hearts,\n",
    "        url,\n",
    "        Email,\n",
    "        timeLike,\n",
    "        # numNum,\n",
    "        # numberWithCommas,\n",
    "        # numComb,\n",
    "        emoticon,\n",
    "        Arrows,\n",
    "        entity,\n",
    "        punctSeq,\n",
    "        arbitraryAbbrev,\n",
    "        separators,\n",
    "        decorations,\n",
    "        embeddedApostrophe,\n",
    "        # Hashtag,\n",
    "        # AtMention,\n",
    "    )), re.UNICODE)\n",
    "\n",
    "# Edge punctuation\n",
    "# Want: 'foo' => ' foo '\n",
    "# While also:   don't => don't\n",
    "# the first is considered \"edge punctuation\".\n",
    "# the second is word-internal punctuation -- don't want to mess with it.\n",
    "# BTO (2011-06): the edgepunct system seems to be the #1 source of problems these days.  \n",
    "# I remember it causing lots of trouble in the past as well.  Would be good to revisit or eliminate.\n",
    "\n",
    "# Note the 'smart quotes' (http://en.wikipedia.org/wiki/Smart_quotes)\n",
    "# edgePunctChars    = r\"'\\\"“”‘’«»{}\\(\\)\\[\\]\\*&\" #add \\\\p{So}? (symbols)\n",
    "edgePunctChars = \"'\\\"“”‘’«»{}\\\\(\\\\)\\\\[\\\\]\\\\*&\"  # add \\\\p{So}? (symbols)\n",
    "edgePunct = \"[\" + edgePunctChars + \"]\"\n",
    "notEdgePunct = \"[a-zA-Z0-9]\"  # content characters\n",
    "offEdge = r\"(^|$|:|;|\\s|\\.|,)\"  # colon here gets \"(hello):\" ==> \"( hello ):\"\n",
    "EdgePunctLeft = re.compile(offEdge + \"(\" + edgePunct + \"+)(\" + notEdgePunct + \")\", re.UNICODE)\n",
    "EdgePunctRight = re.compile(\"(\" + notEdgePunct + \")(\" + edgePunct + \"+)\" + offEdge, re.UNICODE)\n",
    "\n",
    "\n",
    "def splitEdgePunct(text):\n",
    "    text = EdgePunctLeft.sub(r\"\\1\\2 \\3\", text)\n",
    "    text = EdgePunctRight.sub(r\"\\1 \\2\\3\", text)\n",
    "    return text\n",
    "\n",
    "\n",
    "p = parser()\n",
    "info = p.info\n",
    "\n",
    "\n",
    "def dateParse(text):\n",
    "    pat = re.compile(\"(DATE[ ]*)+\", re.UNICODE)\n",
    "    text = re.sub(pat, \"DATE \", text)\n",
    "    return text\n",
    "\n",
    "\n",
    "def timetoken(token):\n",
    "    try:\n",
    "        float(token)\n",
    "        return True\n",
    "    except ValueError:\n",
    "        pass\n",
    "    return any(f(token) for f in\n",
    "               (info.jump, info.weekday, info.month, info.hms, info.ampm, info.pertain, info.utczone, info.tzoffset))\n",
    "\n",
    "\n",
    "def timesplit(input_string):\n",
    "    batch = \"\"\n",
    "    #  for token in _timelex(input_string):\n",
    "    for token in input_string.split():\n",
    "        if timetoken(token):\n",
    "            if info.jump(token):\n",
    "                continue\n",
    "            batch = batch + \"\"\n",
    "        else:\n",
    "            batch = batch + \" \" + token\n",
    "    return dateParse(batch)\n",
    "\n",
    "\n",
    "# timesplit(a)\n",
    "\n",
    "def digitParse(text):\n",
    "    pat = re.compile(\"(DIGIT-DIGIT|DIGIT[ ]*)+\", re.UNICODE)\n",
    "    text = re.sub(pat, \" \", text)\n",
    "    return text\n",
    "\n",
    "\n",
    "def digit(text):\n",
    "    # print (text)\n",
    "    text = re.sub(num, \"\", text)\n",
    "    text = re.sub(numNum, \"\", text)\n",
    "    text = re.sub(numberWithCommas, \"\", text)\n",
    "    text = re.sub(numComb, \"\", text)\n",
    "    return digitParse(text)\n",
    "\n",
    "\n",
    "def urlParse(text):\n",
    "    pat = re.compile(url, re.UNICODE)\n",
    "    text = re.sub(pat, \"\", text)\n",
    "    return text\n",
    "\n",
    "\n",
    "\"\"\"\n",
    " The main work of tokenizing a tweet.\n",
    " Modified by Firoj, for the preprocessing of the crisis data.\n",
    "\"\"\"\n",
    "\n",
    "\n",
    "def simpleTokenize(text):\n",
    "    try:\n",
    "        text = text\n",
    "    except Exception as e:\n",
    "        # print (text)\n",
    "        print(e)\n",
    "        pass\n",
    "    ## 1. Lowercased\n",
    "    text = text.lower();\n",
    "    ## 2. time pattern replaced with  DATE tag\n",
    "    # text=timesplit(text)\n",
    "    ## 3. date pattern replaced with  DIGIT tag    \n",
    "    text = digit(text)\n",
    "    ## 4. URL pattern replaced with  URL tag    \n",
    "    text = urlParse(text)\n",
    "    #\n",
    "    ## 5. Removed special characters and # symbol\n",
    "    punc = \"[#(),$%^&*+={}\\[\\]:\\\"|\\~`<>/,¦!?½£¶¼©⅐⅑⅒⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞⅟↉¤¿º;-]+\"\n",
    "    text = re.sub(punc, \"\", text)\n",
    "    spchar = \"[\\x98\\x9C\\x94\\x89\\x84\\x88\\x92\\x8F]+\"\n",
    "    text = re.sub(spchar, \"\", text)\n",
    "    text = re.sub(\"--&gt;&gt;|--|-|[\\.]+\", \" \", text)\n",
    "    text = re.sub(\"\\'\", \" \", text)\n",
    "    # print (text)\n",
    "\n",
    "    ## 6 and 7. Removed username started with @, single character\n",
    "    tweet_words = text.split(' ')\n",
    "    tWords = []\n",
    "    for word in tweet_words:\n",
    "        word = word.strip()\n",
    "        if ((len(word) > 1 and word[0] == '@')):\n",
    "            continue\n",
    "        elif (word == \"rt\" or word == \"userid\"):\n",
    "            continue\n",
    "        elif (len(word) != 1):\n",
    "            tWords.append(word)\n",
    "    text = \" \".join(tWords)\n",
    "\n",
    "    ## 8. Reduced repeated character     \n",
    "    text = re.sub(r\"(.)\\1\\1+\", r'\\1\\1', text)\n",
    "    splitPunctText = splitEdgePunct(text.strip())\n",
    "    textLength = len(splitPunctText)\n",
    "\n",
    "    # BTO: the logic here got quite convoluted via the Scala porting detour\n",
    "    # It would be good to switch back to a nice simple procedural style like in the Python version\n",
    "    # ... Scala is such a pain.  Never again.\n",
    "\n",
    "    # Find the matches for subsequences that should be protected,\n",
    "    # e.g. URLs, 1.0, U.N.K.L.E., 12:53\n",
    "    bads = []\n",
    "    badSpans = []\n",
    "    for match in Protected.finditer(splitPunctText):\n",
    "        # The spans of the \"bads\" should not be split.\n",
    "        if (match.start() != match.end()):  # unnecessary?\n",
    "            bads.append([splitPunctText[match.start():match.end()]])\n",
    "            badSpans.append((match.start(), match.end()))\n",
    "\n",
    "    # Create a list of indices to create the \"goods\", which can be\n",
    "    # split. We are taking \"bad\" spans like \n",
    "    #     List((2,5), (8,10)) \n",
    "    # to create \n",
    "    #     List(0, 2, 5, 8, 10, 12)\n",
    "    # where, e.g., \"12\" here would be the textLength\n",
    "    # has an even length and no indices are the same\n",
    "    indices = [0]\n",
    "    for (first, second) in badSpans:\n",
    "        indices.append(first)\n",
    "        indices.append(second)\n",
    "    indices.append(textLength)\n",
    "\n",
    "    # Group the indices and map them to their respective portion of the string\n",
    "    splitGoods = []\n",
    "    for i in range(0, len(indices), 2):\n",
    "        goodstr = splitPunctText[indices[i]:indices[i + 1]]\n",
    "        splitstr = goodstr.strip().split(\" \")\n",
    "        splitGoods.append(splitstr)\n",
    "\n",
    "    #  Reinterpolate the 'good' and 'bad' Lists, ensuring that\n",
    "    #  additonal tokens from last good item get included\n",
    "    zippedStr = []\n",
    "    for i in range(len(bads)):\n",
    "        zippedStr = addAllnonempty(zippedStr, splitGoods[i])\n",
    "        zippedStr = addAllnonempty(zippedStr, bads[i])\n",
    "    zippedStr = addAllnonempty(zippedStr, splitGoods[len(bads)])\n",
    "    # text = \" \".join(zippedStr)\n",
    "    return zippedStr  # text #text.split()\n",
    "\n",
    "\n",
    "def addAllnonempty(master, smaller):\n",
    "    for s in smaller:\n",
    "        strim = s.strip()\n",
    "        if (len(strim) > 0):\n",
    "            master.append(strim)\n",
    "    return master\n",
    "\n",
    "\n",
    "# \"foo   bar \" => \"foo bar\"\n",
    "def squeezeWhitespace(input):\n",
    "    return Whitespace.sub(\" \", input).strip()\n",
    "\n",
    "\n",
    "# Final pass tokenization based on special patterns\n",
    "def splitToken(token):\n",
    "    m = Contractions.search(token)\n",
    "    if m:\n",
    "        return [m.group(1), m.group(2)]\n",
    "    return [token]\n",
    "\n",
    "\n",
    "def file_exist(file_name):\n",
    "    if os.path.exists(file_name):\n",
    "        return True\n",
    "    else:\n",
    "        return False\n",
    "\n",
    "\n",
    "def read_stop_words(file_name):\n",
    "    if (not file_exist(file_name)):\n",
    "        print(\"Please check the file for stop words, it is not in provided location \" + file_name)\n",
    "        exit(0)\n",
    "    stop_words = []\n",
    "    with open(file_name, 'r') as f:\n",
    "        for line in f:\n",
    "            line = line.strip()\n",
    "            if (line == \"\"):\n",
    "                continue\n",
    "            stop_words.append(line)\n",
    "    return stop_words;\n",
    "\n",
    "\n",
    "stop_words_file = \"stop_words_english.txt\"\n",
    "stop_words = read_stop_words(stop_words_file)\n",
    "\n",
    "\n",
    "# Assume 'text' has no HTML escaping.\n",
    "def tokenize(text):\n",
    "    # print (text)\n",
    "    text = simpleTokenize(squeezeWhitespace(text));\n",
    "    # print (text)\n",
    "    w_list = []\n",
    "    for w in text:\n",
    "        # print (w)\n",
    "        if w not in stop_words:\n",
    "            try:\n",
    "                w_list.append(w)\n",
    "            except Exception as e:\n",
    "                pass\n",
    "    text = \" \".join(text)\n",
    "    return text\n",
    "\n",
    "\n",
    "# Twitter text comes HTML-escaped, so unescape it.\n",
    "# We also first unescape &amp;'s, in case the text has been buggily double-escaped.\n",
    "def normalizeTextForTagger(text):\n",
    "    text = text.replace(\"&amp;\", \"&\")\n",
    "    text = six.moves.html_parser.HTMLParser().unescape(text)\n",
    "    return text\n",
    "\n",
    "\n",
    "# This is intended for raw tweet text -- we do some HTML entity unescaping before running the tagger.\n",
    "# \n",
    "# This function normalizes the input text BEFORE calling the tokenizer.\n",
    "# So the tokens you get back may not exactly correspond to\n",
    "# substrings of the original text.\n",
    "def tokenizeRawTweetText(text):\n",
    "    tokens = tokenize(normalizeTextForTagger(text))\n",
    "    return tokens\n",
    "\n",
    "\n",
    "\"\"\"\n",
    "Test\n",
    "\"\"\"\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "cae2fffd",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "ded0c660",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
