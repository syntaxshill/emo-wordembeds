import argparse
import gensim
import nltk
import numpy as np
import pickle
import re
from scipy import sparse
from sklearn import dummy, linear_model, naive_bayes, svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support as prf_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics.cluster import contingency_matrix
# from mlxtend.evaluate import mcnemar_table
from statsmodels.stats.contingency_tables import mcnemar
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_SEED = 1509

LABELED_DATA_FN = "train_mk2.pkl"
TEST_DATA_FN = "test_mk2.pkl"
W2V_FN = "word2vec_model_300d.model" #"word2vec_no1.0_model_300d.model"


########################################################################################################################
# Load data from file and process it into matrix form
# data_fn : the name of the .pkl file in which the data may be found
# args : contains arguments for the vectorizer (including which n-grams to use)
# features : a list of additional features to include beyond n-grams
# correlations : a boolean determining whether to calculate Pearson's r for the additional features
def establish_preprocessing(data_fn, args, features, correlations=False):
    with open(data_fn, "rb") as f:
        data = pickle.load(f)

    # filter according to confidence interval
    data = list(filter(lambda d: d['confidence'] >= args.min_confidence, data))

    # gather the data from the pickle file
    X = [' '.join(d['text']).lower() for d in data]

    # word-tokenize the data so punctuation is apart from words
    X = [' '.join(nltk.word_tokenize(d)) for d in X]

    if args.embedding_mode == "ngram":
        # vectorize the data
        vectorizer = CountVectorizer(lowercase=True, max_df=args.max_df, max_features=args.max_features,
                                     ngram_range=(args.min_ngram, args.max_ngram))
        X = vectorizer.fit_transform(X)
    elif args.embedding_mode == "word2vec":
        # we are using different kinds of word2vec files and this try-except should load any of them
        try:
            model = gensim.models.Word2Vec.load(W2V_FN)
        except AttributeError:
            model = gensim.models.KeyedVectors.load(W2V_FN)

        temp_X = []

        # create a sentence vector by averaging all the individual word vectors
        for d in X:
            temp_d = None
            for w in d.split(' '):
                if w in model:  # ignore words not in our vocabulary for now
                    if temp_d is None:
                        temp_d = np.copy(model[w])
                    else:
                        temp_d += model[w]

            # take the average word embedding
            temp_X.append(temp_d / len(d.split(' ')))

        X = np.asarray(temp_X)

    else:
        raise ValueError("Unsupported embedding style. Supported embedding styles: ngram, word2vec")

    print("Data size is", X.shape[0])
    y = [d['label'] for d in data]

    # add non-n-gram features
    extra_feats = []
    if "timestamps" in features:
        times = np.array([d['timestamp'] for d in data])
        extra_feats.append(times)

    dal_features = []
    for feature in list(filter(lambda x: "dal" in x, features)):
        this_feat = np.array([d[feature] for d in data])
        dal_features.append((feature, this_feat))
        extra_feats.append(this_feat)

    if "sentiment" in features:
        sentiments = np.array([d['sentiment_polarity'] for d in data])
        extra_feats.append(sentiments)

    liwc_features = []
    for feature in list(filter(lambda x: "liwc" in x, features)):
        this_feat = np.array([d[feature] for d in data])
        liwc_features.append((feature, this_feat))
        extra_feats.append(this_feat)


    def find_liwc_string(search, target):
        results = []
        if "*" not in target:
            # look for the word flanked by string start/end, whitespace, or punctuation
            for m in re.finditer("(?:\s|\A|[.,?!:;()\[\]'\"`])" + target + "(?:\s|\Z|[.,?!:;()\[\]'\"`])", search):
                results.append(m.start(0))
        else:
            # look for the word (minus the *) and any number of ending characters flanked by the above
            for m in re.finditer("(?:\s|\A|[.,?!:;()\[\]'\"`])" + target[:-1] + "\S*(?:\s|\Z|[.,?!:;()\[\]'\"`])", search):
                results.append(m.start(0))
        if len(results) != 0:
            return results[0]
        else:
            return -1

    #fear + pronouns
    if "fear" in features:
        first_person = ['I', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']
        third_person = ['he', 'she', 'him', 'her', 'his', 'her', 'hers',
                        'they', 'them', 'their', 'theirs']
        fear_fp_feat = []
        fear_tp_feat = []
        fear = ['abandon*', 'agonize', 'agony', 'alone', 'bereave*', 'broke', 'cried', 'cries', 'crushed', 'cry', 'crying',
                'depress*', 'deprive', 'despair*', 'devastate', 'disappoint*', 'discourage', 'dishearten*', 'doom*', 'dull',
                'emptier', 'emptiest', 'emptiness', 'empty', 'fail*', 'fatigue', 'flunk*', 'gloom', 'gloomier', 'gloomiest',
                'gloomily', 'gloominess', 'gloomy', 'grave*', 'grief', 'griey*', 'grim', 'grimly', 'heartbreak*',
                'heartbroke*', 'helpless*', 'homesick*', 'hopeless*', 'hurt*', 'inadequa*', 'inferior', 'inferiority',
                'isolat*', 'lame', 'lamely', 'lameness', 'lamer', 'lamest', 'lone', 'lonelier', 'loneliest', 'loneliness',
                'lonely', 'loner*', 'longing*', 'lose', 'loser*', 'loses', 'losing', 'loss*', 'lost', 'low', 'lower',
                'Lowered', 'lowering', 'lowers', 'lowest', 'lowli*', 'lowly', 'melanchol*', 'miser*', 'miss', 'missed',
                'misses', 'missing', 'mourn*', 'neglect*', 'overwhelm*', 'pathetic', 'pathetically', "pessimis'", 'pitiable',
                'pitied', 'pities', 'pitiful', 'pitifully', 'pity*', 'regret*', 'reject*', 'remorse*', 'resign*', 'ruin*',
                'sad', 'sadder', 'saddest', 'sadly', 'sadness', 'sigh', 'sighed', 'sighing', 'sighs', 'sob', 'sobbed',
                'sobbing', 'sobs', 'solemn*', 'sorrow*', 'sorrow*', 'sorry', 'suffer', 'suffered', 'sufferer*', 'suffering',
                'suffers', 'tears', 'traged*', 'tragic', 'tragically', 'unhap*', 'unimportant', 'unsuccessful*', 'useless',
                'uselessly', 'uselessness', 'weep', 'wept', 'whine*', 'whining', 'woe*', 'worthless', 'yearn*']
        for d in data:
            z = nltk.word_tokenize(' '.join(d['text']))
            words = len(z)
            passed_dist = 0
            distance = 0
            count = 0
            for i in range(0, words):
                passed_dist += len(z[i])
                if z[i] in first_person:
                    for ang in fear:
                        distances = []
                        for j in range(0, len(d['text'])):
                            dist = find_liwc_string(d['text'][j], ang)
                            if dist > i:
                                distances.append(dist - passed_dist)
                            elif dist > 0:
                                distances.append(passed_dist - dist)
                        if len(distances) != 0:
                            distance += min(distances)
                            count += 1
            avg_dist = 1000000
            if count != 0:
                avg_dist = distance / count
            fear_fp_feat.append(avg_dist)
        fear_fp_feat = np.array(fear_fp_feat)
        extra_feats.append(fear_fp_feat)

        for d in data:
            z = nltk.word_tokenize(' '.join(d['text']))
            words = len(z)
            passed_dist = 0
            distance = 0
            count = 0
            for i in range(0, words):
                passed_dist += len(z[i])
                if z[i] in third_person:
                    for ang in fear:
                        distances = []
                        for j in range(0, len(d['text'])):
                            dist = find_liwc_string(d['text'][j], ang)
                            if dist > i:
                                distances.append(dist - passed_dist)
                            elif dist > 0:
                                distances.append(passed_dist - dist)
                        if len(distances) != 0:
                            distance += min(distances)
                            count += 1
            avg_dist = 1000000
            if count != 0:
                avg_dist = distance / count
            fear_tp_feat.append(avg_dist)

        ang_tp_feat = np.array(fear_tp_feat)
        extra_feats.append(ang_tp_feat)


    # anger + pronouns
    if "anger" in features:
        first_person = ['I', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']
        third_person = ['he', 'she', 'him', 'her', 'his', 'her', 'hers',
                        'they', 'them', 'their', 'theirs']
        ang_fp_feat = []
        ang_tp_feat = []

        angry = ['abuse*', 'poison*', 'abusi*', 'prejudic*', 'aggravat*', 'prick*', 'aggress', 'protest', 'aggressed',
                 'protested', 'aggresses', 'protesting', 'aggressing', 'protests', 'aggression*', 'punish*', 'aggressive',
                 'pushy', 'aggressively', 'rage*', 'aggressor*', 'raging', 'agitat*', 'rape*', 'anger*', 'raping',
                 'angrier', 'rapist*', 'angriest', 'rebel*', 'angry', 'resent*', 'annoy', 'revenge*', 'annoyed', 'ridicul*',
                 'annoying', 'rude', 'annoys', 'rudely', 'antagoni*', 'sarcas*', 'argil*', 'savage*', 'argu*', 'sceptic*',
                 'assault*', 'screw*', 'asshole*', 'shit*', 'attack*', 'sinister', 'bastard*', 'smother*', 'battl*',
                 'snob*', 'beaten', 'spite*', 'bitch*', 'stubborn*', 'bitter', 'stupid', 'bitterly', 'stupider',
                 'bitterness', 'stupidest', 'blam*', 'stupidity', 'bother*', 'stupidly', 'brutal*', 'suck', 'cheat*',
                 'sucked', 'confront*', 'sucker*', 'contempt*', 'sucks', 'contempt*', 'sucks', 'contradic*', 'sucky',
                 'crap', 'tantrum*', 'crappy', 'teas*', 'critical', 'temper', 'critici*', 'tempers', 'cruel', 'threat*',
                 'crueler', 'tortur*', 'cruelest', 'trick', 'cruelty', 'tricked', 'cunt*', 'trickier', 'cynic*', 'trickiest',
                 'damn*', 'tricks', 'despise', 'tricky', 'destroy*', 'uglier', 'destruct', 'ugliest', 'destructed', 'ugly',
                 'vicious', 'destructive', 'viciously', 'destructivness', 'viciousness', 'distrust*', 'vile', 'distrust',
                 'villain*', 'dumb', 'violat*', 'dumbass*', 'violent', 'destruction', 'dumber', 'violently', 'dumbest',
                 'war', 'dummy', 'warfare*', 'enemie*', 'warred', 'enemy*', 'warring', 'enrag*', 'wars', 'envie*',
                 'weapon*', 'envious', 'wicked', 'envy*', 'wickedly', 'feroc*', 'yell', 'feud*', 'yelled', 'fiery', 'yelling',
                 'fight*', 'yells', 'foe*', 'fought', 'frustrat*', 'tuck', 'tucked*', 'fucker*', 'fuckface*', 'fuckh*',
                 'fuckin*', 'fucks', 'fucktard', 'fucktwat', 'fuckwad', 'fume*', 'fuming', 'furious*', 'fury', 'goddam*',
                 'greed*', 'grouch*', 'grr*', 'grudg*', 'harass*', 'hate', 'hated', 'hateful*', 'hater*', 'hates',
                 'hating', 'hatred', 'heartless*', 'hell', 'hellish', 'hostil*', 'humiliat*', 'idiot*', 'insult*', 'interrup*',
                 'intimidat*', 'jealous', 'jealousies', 'jealously','jealousy', 'jerk', 'jerked', 'jerks', 'liar*', 'lied',
                 'lies', 'ludicrous*', 'lying', 'mad', 'maddening*', 'madder', 'maddest', 'maniac*', 'meaner', 'meanest',
                 'mock', 'mocked', 'mocker*', 'mocking', 'mocks', 'moron*', 'murder*', 'nag*', 'nast*', 'obnoxious*',
                 'offence*', 'offend*', 'offense', 'offenses', 'offensive', 'outrag*', 'pest*', 'pettier', 'pettiest',
                 'petty', 'piss*', '']


        for d in data:
            z = nltk.word_tokenize(' '.join(d['text']))
            words = len(z)
            passed_dist = 0
            distance = 0
            count = 0
            for i in range(0, words):
                passed_dist += len(z[i])
                if z[i] in first_person:
                    for ang in angry:
                        distances = []
                        for j in range(0, len(d['text'])):
                            dist = find_liwc_string(d['text'][j], ang)
                            if dist > i:
                                distances.append(dist - passed_dist)
                            elif dist > 0:
                                distances.append(passed_dist - dist)
                        if len(distances) != 0:
                            distance += min(distances)
                            count += 1
            avg_dist = 1000000
            if count != 0:
                avg_dist = distance / count
            ang_fp_feat.append(avg_dist)
        ang_fp_feat = np.array(ang_fp_feat)
        extra_feats.append(ang_fp_feat)

        for d in data:
            z = nltk.word_tokenize(' '.join(d['text']))
            words = len(z)
            passed_dist = 0
            distance = 0
            count = 0
            for i in range(0, words):
                passed_dist += len(z[i])
                if z[i] in third_person:
                    for ang in angry:
                        distances = []
                        for j in range(0, len(d['text'])):
                            dist = find_liwc_string(d['text'][j], ang)
                            if dist > i:
                                distances.append(dist - passed_dist)
                            elif dist > 0:
                                distances.append(passed_dist - dist)
                        if len(distances) != 0:
                            distance += min(distances)
                            count += 1
            avg_dist = 1000000
            if count != 0:
                avg_dist = distance / count
            ang_tp_feat.append(avg_dist)

        ang_tp_feat = np.array(ang_tp_feat)
        extra_feats.append(ang_tp_feat)

    # pronouns
    if "pronouns" in features:
        first_person = ['I', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']
        third_person = ['he', 'she', 'him', 'her', 'his', 'her', 'hers',
                        'they', 'them', 'their', 'theirs']
        fp_feat = []
        tp_feat = []

        emotion_words = ["afraid", "alarm*", "anxiety", "anxious", "anxiously", "anxiousness", "apprehens*", "asham*",
                         "aversi*", "avoid*", "awkward", "confuse", "confused", "confusedly", "confusing", "desperat*",
                         "discomfort*", "distraught", "distress*", "disturb*", "doubt*", "dread*", "dwell*",
                         "embarrass*", "fear", "feared", "fearful*", "fearing", "fears", "frantic*", "fright*", "guilt",
                         "guilt-trip", "guilty", "hesita*", "horrible", "horriby", "horrid*", "horror*", "humiliat*",
                         "impatien*", "inadequa*", "indecis*", "inhibit*", "insecur*", "irrational*", "irrita*",
                         "miser*", "nervous", "nervously", "nervousness", "neurotic*", "obsess*", "overwhelm*", "panic*",
                         "paranoi*", "petrif*", "phobi*", "pressur*", "reluctan*", "repress*", "restless*", "rigid",
                         "rigidity", "rigidly", "risk*", "scare", "scared", "scares", "scarier", "scariest", "scaring",
                         "scary", "shake*", "shaki*", "shaky", "shame*", "shook", "shy", "shyly", "shyness", "startl*",
                         "strain*", "stress*", "struggl*", "suspicio*", "tense", "tensely", "tensing", "tension*",
                         "terrified", "terrifies", "terrify", "terrifying", "terror*", "threat*", "timid*", "trembl*",
                         "turmoil", "twitchy", "uncertain*", "uncomfortabl*", "uncontrol*", "uneas*", "unsettl*",
                         "unsure*", "upset", "upsets", "upsetting", "uptight*", "vulnerab*", "worried", "worrier",
                         "worries", "worry", "worrying", "abandon*", "abuse*", "abusi*", "ache*", "aching*", "advers*",
                         "afraid", "aggravat*", "aggress", "aggressed", "aggresses", "aggressing", "aggression*",
                         "aggressive", "aggressively", "aggressor*", "agitat*", "agoniz*", "agony", "alarm*", "alone",
                         "anger*", "angrier", "angriest", "angry", "anguish*", "annoy", "annoyed", "annoying", "annoys",
                         "antagoni*", "anxiety", "anxious", "anxiously", "anxiousness", "apath*", "appall*",
                         "apprehens*", "argh*", "argu*", "arrogan*", "asham*", "assault*", "attack*", "avers*",
                         "avoid*", "awful", "awkward", "bad", "badly", "bashful*", "bastard*", "battl*", "beaten",
                         "bereave*", "bitch*", "disliking", "dismay*", "disreput*", "diss", "dissatisf*","distraught",
                         "distress*", "distrust*", "disturb*", "domina*", "doom*", "dork*", "doubt*", "dread*", "dull",
                         "dumb", "dumbass*", "dumber", "dumbest", "dummy", "dump*", "dwell*", "egotis*", "embarrass*",
                         "emotional","emptier", "emptiest", "emptiness", "empty", "enemie*", "enemy*", "enrag*",
                         "envie*", "envious","envy*", "evil", "excruciat*", "exhaust*", "fail*", "fake", "fatal*",
                         "fatigu*", "fault*", "fear", "feared", "fearful*", "fearing", "fears", "feroc*", "feud*",
                         "fiery", "fight*", "fired", "flunk*", "fool", "fooled", "fooling", "foolish", "ignoramus",
                         "ignorant", "ignore", "ignored", "ignores", "ignoring", "immoral*", "impatien*", "impersonal",
                         "impolite*", "inadequa*", "incompeten*", "indecis*", "ineffect*", "inferior", "inferiority",
                         "inhibit*", "insecur*", "insincer*", "insult*", "interrup*", "intimidat*", "irrational*",
                         "irrita*", "isolat*", "jaded", "jealous", "jealousies", "jealously", "jealousy", "jerk",
                         "jerked", "jerks", "kill*", "lame", "lamely", "lameness", "lamer", "lamest", "lazier",
                         "laziest", "lazy", "liabilit*", "liar*", "lied", "lies", "lone", "lonelier", "loneliest",
                         "loneliness", "lonely", "loner*", "longing*", "lose", "loser*", "loses", "losing", "lost",
                         "poorer", "poorest", "poorly", "poorness*", "powerless*", "prejudic*", "pressur*", "prick*",
                         "problem*", "protest", "protested", "protesting", "protests", "puk*", "punish*", "pushy",
                         "queas*", "rage*", "raging", "rancid*", "rape*", "raping", "rapist*", "rebel*", "reek*",
                         "regret*", "reject*", "reluctan*", "remorse*", "repress*", "resent*", "resign*", "restless*",
                         "revenge*", "ridicul*", "rigid", "rigidity", "rigidly", "risk*", "rotten", "rude", "rudely",
                         "rum*", "sad", "sadder", "saddest", "sadly", "sadness", "sarcas*", "savage*", "scare",
                         "scared", "scares", "scarier", "scariest", "scaring", "scary", "sceptic*", "scream*", "tragic",
                         "tragically", "trauma*", "trembl*", "trick", "tricked", "trickier", "trickiest", "tricks",
                         "tricky", "trite", "trivial", "troubl*", "turmoil", "twitchy", "ugh", "uglier", "ugliest",
                         "ugly", "unaccept*", "unattractive", "uncertain*", "uncomfortabl*", "uncontrol*", "undesir*",
                         "uneas*", "unfair", "unfortunate*", "unfriendly", "ungrateful*", "unhapp*", "unimportant",
                         "unimpress*", "unkind", "unlov*", "unlucky", "unpleasant", "unprotected", "unsafe", "unsavory",
                         "unsettl*", "unsuccessful*", "unsure*", "unwelcom*", "upset", "upsets", "upsetting",
                         "uptight*", "useless", "uselessly", "uselessness", "vain", "vanity", "vicious", "viciously",
                         "viciousness", "victim*", "vile", "villain*", "bitter", "bitterly", "bitterness", "blam*",
                         "bore*", "boring", "bother*", "broke", "brutal*", "burden*", "careless*", "cheat*", "coldly",
                         "complain*", "condemn*", "confront*", "confuse", "confused", "confusedly", "confusing",
                         "contempt*", "contradic*", "crap", "crappy", "crazy", "cried", "cries", "critical", "critici*",
                         "crude", "crudely", "cruel", "crueler", "cruelest", "cruelty", "crushed", "cry", "crying",
                         "cunt*", "curse", "cut", "cynic*", "damag*", "damn*", "danger", "dangerous", "dangerously",
                         "dangers", "daze*", "decay*", "deceptive", "deceiv*", "defeat*", "defect*", "defenc*",
                         "defend*", "defense", "defenseless", "defensive", "defensively", "defensiveness", "foolishly",
                         "fools", "forbade", "forbid", "forbidden", "forbidding", "forbids", "fought", "frantic*",
                         "freak*", "fright*", "frustrat*", "fuck", "fucked*", "fucker*", "fuckface*", "fuckh*",
                         "fuckin*", "fucks", "fucktard", "fucktwat*", "fuckwad*", "fume*", "fuming", "furious*", "fury",
                         "geek*", "gloom", "gloomier", "gloomiest", "gloomily", "gloominess", "gloomy", "goddam*",
                         "good-for-nothing", "gossip*", "grave*", "greed*", "grief", "griev*", "grim", "grimac*",
                         "grimly", "gross", "grossed", "grosser", "grossest", "grossing", "grossly", "grossness",
                         "grouch*", "grr*", "grudg*", "guilt", "guilt-trip*", "guiltier", "guiltiest", "guilty",
                         "hangover*", "harass*", "harm", "lous*", "loveless", "low", "lower", "lowered", "lowering",
                         "lowers", "lowest", "lowli*", "lowly", "luckless*", "ludicrous*", "lying", "mad", "maddening*",
                         "madder", "maddest", "maniac*", "masochis*", "meaner", "meanest", "melanchol*", "mess",
                         "messier", "messiest", "messy", "miser*", "miss", "missed", "misses", "missing", "mistak*",
                         "mock", "mocked", "mocker*", "mocking", "mocks", "molest*", "mooch*", "moodi*", "moody",
                         "moron*", "mourn*", "murder*", "nag*", "nast*", "needy", "neglect*", "nerd*", "nervous",
                         "nervously", "nervousness", "neurotic*", "nightmar*", "numbed", "numbing", "numbness", "numb*",
                         "obnoxious*", "obsess*", "offence*", "screw*", "selfish*", "serious", "seriously",
                         "seriousness", "severe*", "shake*", "shaki*", "shaky", "shame*", "shit*", "shock*", "shook",
                         "shy", "shyly", "shyness", "sick", "sicken*", "sicker", "sickest", "sickly", "sigh", "sighed",
                         "sighing", "sighs", "sin", "sinister", "sins", "slut*", "smh", "smother*", "smug*", "snob*",
                         "sob", "sobbed", "sobbing", "sobs", "solemn*", "sorrow*", "sorry", "spite*", "stale",
                         "stammer*", "stank*", "startl*", "steal*", "stench*", "stink", "stinky", "strain*", "strange",
                         "strangest", "stress*", "struggl*", "stubborn*", "stunk", "stupid", "stupider", "stupidest",
                         "stupidity", "stupidly", "violat*", "violence", "violent", "violently", "vomit*", "vulnerab*",
                         "war", "warfare*", "warn*", "warred", "warring", "wars", "weak", "weaken", "weakened",
                         "weakening", "weakens", "weaker", "weakest", "weakling", "weakly", "weapon*", "weary", "weep*",
                         "weird", "weirded", "weirder", "weirdest", "weirdly", "weirdness", "weirdo", "weirdos",
                         "weirds", "wept", "whine*", "whining", "whore*", "wicked", "wickedly", "wimp*", "witch*",
                         "woe*", "worried", "worrier", "worries", "worry", "worrying", "worse", "worsen", "worsened",
                         "worsening", "worsens", "worst", "worthless", "wrong", "wrongdoing", "wronged", "wrongful",
                         "wrongly", "wrongness", "wrongs", "degrad*", "demean*", "demot*", "denial", "depress*",
                         "depriv*", "despair*", "desperat*", "despis*", "destroy*", "destruct", "destructed",
                         "destruction", "destructive", "destructivness", "devastat*", "devensiveness", "devil*",
                         "difficult", "difficulties", "difficulty", "disadvantag*", "disagree*", "disappoint*",
                         "disaster*", "discomfort*", "discourag*", "disgrac*", "disgust*", "dishearten*", "dishonor*",
                         "disillusion*", "dislike", "disliked", "dislikes", "harmed", "harmful", "harmfully",
                         "harmfulness", "harming", "harms", "harsh", "hate", "hated", "hateful*", "hater*", "hates",
                         "hating", "hatred", "haunted", "hazard*", "heartbreak*", "heartbroke*", "heartless*", "hell",
                         "hellish", "helpless*", "hesita*", "homesick*", "hopeless*", "horrible", "horribly", "horrid*",
                         "horror*", "hostil*", "humiliat*", "hungover", "hurt*", "idiot*", "ignorable", "offend*",
                         "offense", "offenses", "offensive", "outrag*", "overwhelm*", "pain", "pained", "painf*",
                         "pains", "pamc*", "paranm*", "pathetic", "pathetically", "peculiar*", "perv", "perver*",
                         "pervy", "pessims*", "pest*", "petrif*", "pettier", "pettiest", "petty", "phobi*", "phony",
                         "piss*", "pitiable", "pitied", "pities", "pitiful", "pitifully", "pity*", "poison*", "poor",
                         "stutter*", "suck", "sucked", "sucker*", "sucks", "sucky", "suffer", "suffered", "sufferer*",
                         "suffering", "suffers", "suspicio*", "tantrum*", "tears", "teas*", "tedious", "temper",
                         "tempers", "tense", "tensely", "tensing", "tension*", "terrible", "terribly", "terrified",
                         "terrifies", "terrify", "terrifying", "terror*", "thief", "thiev*", "threat*", "timid*",
                         "tortur*", "traged*", "yearn*", "yell", "yelled", "yelling", "yells", "yuck"]
        for d in data:
            z = nltk.word_tokenize(' '.join(d['text']))
            words = len(z)
            passed_dist = 0
            distance = 0
            count = 0
            for i in range(0, words):
                passed_dist += len(z[i])
                if z[i] in first_person:
                    for sad in emotion_words:
                        distances = []
                        for j in range(0, len(d['text'])):
                            dist = find_liwc_string(d['text'][j], sad)
                            if dist > passed_dist:
                                distances.append(dist - passed_dist)
                            elif dist > 0:
                                distances.append(passed_dist - dist)
                        if len(distances) != 0:
                            distance += min(distances)
                            count += 1
            avg_dist = 1000000
            if count != 0:
                avg_dist = distance / count
            fp_feat.append(avg_dist)
        fp_feat = np.array(fp_feat)
        extra_feats.append(fp_feat)

        for d in data:
            z = nltk.word_tokenize(' '.join(d['text']))
            words = len(z)
            passed_dist = 0
            distance = 0
            count = 0
            for i in range(0, words):
                passed_dist += len(z[i])
                if z[i] in third_person:
                    for sad in emotion_words:
                        distances = []
                        for j in range(0, len(d['text'])):
                            dist = find_liwc_string(d['text'][j], sad)
                            if dist > passed_dist:
                                distances.append(dist - passed_dist)
                            elif dist > 0:
                                distances.append(passed_dist - dist)
                        if len(distances) != 0:
                            distance += min(distances)
                            count += 1
            avg_dist = 1000000
            if count != 0:
                avg_dist = distance / count
            tp_feat.append(avg_dist)

        tp_feat = np.array(tp_feat)
        extra_feats.append(tp_feat)

        # pronoun count
        # for d in data:
        #     f_count = 0
        #     t_count = 0
        #     z = [nltk.word_tokenize(' '.join(d['text']))]
        #     for word in z:
        #         for first in first_person:
        #             if word == first:
        #                 f_count += 1
        #         for third in third_person:
        #             if word == third:
        #                 t_count += 1
        #     fp_feat.append(f_count)
        #     tp_feat.append(t_count)

    # # Diana's attempt
    # if "dianas_feat" in features:
    #     stress_words = ['anxious', 'bad', 'ashamed', 'stressed', 'sick', 'enraged', 'weak', 'terrified', 'shitty',
    #                     'lost',
    #                     'alone', 'anxious', 'worse', 'crappy', 'sad', 'lonely', 'upset', 'sad', 'afraid', 'scared']
    #     my_feat = []
    #     found = False
    #     for d in data:
    #         z = [nltk.word_tokenize(' '.join(d['text']))]
    #         i = 0
    #         for word in z:
    #             if word == 'I' and z[i + 1] == 'feel':
    #                 for stress in stress_words:
    #                     if z[i + 2] == stress:
    #                         my_feat.append(1)
    #                         found = True
    #             if not found:
    #                 my_feat.append(0)
    #             i += 1
    #     my_feat = np.array(my_feat)
    #     extra_feats.append(my_feat)

    # now run correlation on these features and the labels
    if correlations and len(extra_feats) > 0:
        print("Calculating correlation coefficients (Pearson's r)...")

        # if "timestamps" in features:
        #     print("Posting time:", np.corrcoef(y, times)[1, 0])
        #
        # for feature in dal_features:
        #     print(feature[0], np.corrcoef(y, feature[1])[1, 0])
        #
        # if "sentiment" in features:
        #     print("Sentiment:", np.corrcoef(y, sentiments)[1, 0])
        #
        for feature in liwc_features:
            if feature[0] == 'liwc_i':
                print("proximity and liwc_i: ", np.corrcoef(fp_feat, feature[1])[1, 0])
            if feature[0] == 'liwc_negemo':
                print("proximity and liwc_negemo: ", np.corrcoef(fp_feat, feature[1])[1, 0])
        #     print(feature[0], np.corrcoef(y, feature[1])[1, 0])
        #
        # if "pronouns" in features:
        #     print("First-Person Pronouns:", np.corrcoef(y, fp_feat)[1, 0])
        #     print("Third-Person Pronouns:", np.corrcoef(y, tp_feat)[1, 0])
        #
        # if "fear" in features:
        #     print("First-Person Pronouns (fear):", np.corrcoef(y, fear_fp_feat)[1, 0])
        #     print("Third-Person Pronouns (fear):", np.corrcoef(y, fear_tp_feat)[1, 0])
        #
        # if "anger" in features:
        #     print("First-Person Pronouns (anger):", np.corrcoef(y, ang_fp_feat)[1, 0])
        #     print("Third-Person Pronouns (anger):", np.corrcoef(y, ang_tp_feat)[1, 0])

    scaler = MaxAbsScaler(copy=False)

    # add on the extra features, scale things into the same space, and return the data
    if len(extra_feats) > 0:
        extra_feats = np.vstack(extra_feats).transpose()
        if sparse.issparse(X):  # the n-gram vectorizer outputs a sparse matrix; the word embeddings do not
            # we want to scale the extra features so they are all in the same space, but can't scale ngram counts
            extra_feats = scaler.fit_transform(extra_feats)
            X = np.hstack((X.todense(), extra_feats))
            if args.sparse:
                X = sparse.csr_matrix(X)
        else:
            # but we can scale word embeddings! hopefully this is a good idea
            X = np.hstack((X, extra_feats))
            X = scaler.fit_transform(X)
    elif sparse.issparse(X) and not args.sparse:
        X = X.todense()

    if args.embedding_mode == 'ngram':
        return X, y, scaler, vectorizer
    else:
        return X, y, scaler


########################################################################################################################
# Load data from file and process it into matrix form, given an established preprocessor (most important for BOW)
# data : a list of data to be processed (rather than a file, as above)
# args : contains arguments for the vectorizer (including which n-grams to use)
# features : a list of additional features to include beyond n-grams
# labels : a boolean determining whether the data also include labels (e.g., a test set vs unlabeled data)
def preprocess_data(data, scaler, args, features, vectorizer=None, labels=False):
    # gather the text data
    X = [' '.join(d['text']).lower() for d in data]

    # word-tokenize the data so punctuation is apart from words
    X = [' '.join(nltk.word_tokenize(d)) for d in X]

    if args.embedding_mode == "ngram":
        X = vectorizer.transform(X)
    elif args.embedding_mode == "word2vec":
        try:
            model = gensim.models.Word2Vec.load(W2V_FN)
        except AttributeError:
            model = gensim.models.KeyedVectors.load(W2V_FN)

        temp_X = []

        # create a sentence vector by averaging all the individual word vectors
        for d in X:
            temp_d = None
            for w in d.split(' '):
                if w in model:  # ignore words not in our vocabulary for now
                    if temp_d is None:
                        temp_d = np.copy(model[w])
                    else:
                        temp_d += model[w]

            # take the average word embedding
            temp_X.append(temp_d / len(d.split(' ')))

        X = np.asarray(temp_X)

    else:
        raise ValueError("Unsupported embedding style. Supported embedding styles: word2vec")

    print("Data size is", X.shape[0])

    # add non-n-gram features
    extra_feats = []
    if "timestamps" in features:
        times = np.array([d['timestamp'] for d in data])
        extra_feats.append(times)

    dal_features = []
    for feature in list(filter(lambda x: "dal" in x, features)):
        this_feat = np.array([d[feature] for d in data])
        dal_features.append((feature, this_feat))
        extra_feats.append(this_feat)

    if "sentiment" in features:
        sentiments = np.array([d['sentiment_polarity'] for d in data])
        extra_feats.append(sentiments)

    liwc_features = []
    for feature in list(filter(lambda x: "liwc" in x, features)):
        this_feat = np.array([d[feature] for d in data])
        liwc_features.append((feature, this_feat))
        extra_feats.append(this_feat)


    def find_liwc_string(search, target):
        results = []
        if "*" not in target:
            # look for the word flanked by string start/end, whitespace, or punctuation
            for m in re.finditer("(?:\s|\A|[.,?!:;()\[\]'\"`])" + target + "(?:\s|\Z|[.,?!:;()\[\]'\"`])", search):
                results.append(m.start(0))
        else:
            # look for the word (minus the *) and any number of ending characters flanked by the above
            for m in re.finditer("(?:\s|\A|[.,?!:;()\[\]'\"`])" + target[:-1] + "\S*(?:\s|\Z|[.,?!:;()\[\]'\"`])", search):
                results.append(m.start(0))
        if len(results) != 0:
            return results[0]
        else:
            return -1
    #fear + pronouns
    if "fear" in features:
        first_person = ['I', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']
        third_person = ['he', 'she', 'him', 'her', 'his', 'her', 'hers',
                        'they', 'them', 'their', 'theirs']
        fear_fp_feat = []
        fear_tp_feat = []
        fear = ['abandon*', 'agonize', 'agony', 'alone', 'bereave*', 'broke', 'cried', 'cries', 'crushed', 'cry', 'crying',
                'depress*', 'deprive', 'despair*', 'devastate', 'disappoint*', 'discourage', 'dishearten*', 'doom*', 'dull',
                'emptier', 'emptiest', 'emptiness', 'empty', 'fail*', 'fatigue', 'flunk*', 'gloom', 'gloomier', 'gloomiest',
                'gloomily', 'gloominess', 'gloomy', 'grave*', 'grief', 'griey*', 'grim', 'grimly', 'heartbreak*',
                'heartbroke*', 'helpless*', 'homesick*', 'hopeless*', 'hurt*', 'inadequa*', 'inferior', 'inferiority',
                'isolat*', 'lame', 'lamely', 'lameness', 'lamer', 'lamest', 'lone', 'lonelier', 'loneliest', 'loneliness',
                'lonely', 'loner*', 'longing*', 'lose', 'loser*', 'loses', 'losing', 'loss*', 'lost', 'low', 'lower',
                'Lowered', 'lowering', 'lowers', 'lowest', 'lowli*', 'lowly', 'melanchol*', 'miser*', 'miss', 'missed',
                'misses', 'missing', 'mourn*', 'neglect*', 'overwhelm*', 'pathetic', 'pathetically', "pessimis'", 'pitiable',
                'pitied', 'pities', 'pitiful', 'pitifully', 'pity*', 'regret*', 'reject*', 'remorse*', 'resign*', 'ruin*',
                'sad', 'sadder', 'saddest', 'sadly', 'sadness', 'sigh', 'sighed', 'sighing', 'sighs', 'sob', 'sobbed',
                'sobbing', 'sobs', 'solemn*', 'sorrow*', 'sorrow*', 'sorry', 'suffer', 'suffered', 'sufferer*', 'suffering',
                'suffers', 'tears', 'traged*', 'tragic', 'tragically', 'unhap*', 'unimportant', 'unsuccessful*', 'useless',
                'uselessly', 'uselessness', 'weep', 'wept', 'whine*', 'whining', 'woe*', 'worthless', 'yearn*']
        for d in data:
            z = nltk.word_tokenize(' '.join(d['text']))
            words = len(z)
            passed_dist = 0
            distance = 0
            count = 0
            for i in range(0, words):
                passed_dist += len(z[i])
                if z[i] in first_person:
                    for ang in fear:
                        distances = []
                        for j in range(0, len(d['text'])):
                            dist = find_liwc_string(d['text'][j], ang)
                            if dist > i:
                                distances.append(dist - passed_dist)
                            elif dist > 0:
                                distances.append(passed_dist - dist)
                        if len(distances) != 0:
                            distance += min(distances)
                            count += 1
            avg_dist = 1000000
            if count != 0:
                avg_dist = distance / count
            fear_fp_feat.append(avg_dist)
        fear_fp_feat = np.array(fear_fp_feat)
        extra_feats.append(fear_fp_feat)

        for d in data:
            z = nltk.word_tokenize(' '.join(d['text']))
            words = len(z)
            passed_dist = 0
            distance = 0
            count = 0
            for i in range(0, words):
                passed_dist += len(z[i])
                if z[i] in third_person:
                    for ang in fear:
                        distances = []
                        for j in range(0, len(d['text'])):
                            dist = find_liwc_string(d['text'][j], ang)
                            if dist > i:
                                distances.append(dist - passed_dist)
                            elif dist > 0:
                                distances.append(passed_dist - dist)
                        if len(distances) != 0:
                            distance += min(distances)
                            count += 1
            avg_dist = 1000000
            if count != 0:
                avg_dist = distance / count
            fear_tp_feat.append(avg_dist)

        ang_tp_feat = np.array(fear_tp_feat)
        extra_feats.append(ang_tp_feat)


    # anger + pronouns
    if "anger" in features:
        first_person = ['I', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']
        third_person = ['he', 'she', 'him', 'her', 'his', 'her', 'hers',
                        'they', 'them', 'their', 'theirs']
        ang_fp_feat = []
        ang_tp_feat = []

        angry = ['abuse*', 'poison*', 'abusi*', 'prejudic*', 'aggravat*', 'prick*', 'aggress', 'protest', 'aggressed',
                 'protested', 'aggresses', 'protesting', 'aggressing', 'protests', 'aggression*', 'punish*', 'aggressive',
                 'pushy', 'aggressively', 'rage*', 'aggressor*', 'raging', 'agitat*', 'rape*', 'anger*', 'raping',
                 'angrier', 'rapist*', 'angriest', 'rebel*', 'angry', 'resent*', 'annoy', 'revenge*', 'annoyed', 'ridicul*',
                 'annoying', 'rude', 'annoys', 'rudely', 'antagoni*', 'sarcas*', 'argil*', 'savage*', 'argu*', 'sceptic*',
                 'assault*', 'screw*', 'asshole*', 'shit*', 'attack*', 'sinister', 'bastard*', 'smother*', 'battl*',
                 'snob*', 'beaten', 'spite*', 'bitch*', 'stubborn*', 'bitter', 'stupid', 'bitterly', 'stupider',
                 'bitterness', 'stupidest', 'blam*', 'stupidity', 'bother*', 'stupidly', 'brutal*', 'suck', 'cheat*',
                 'sucked', 'confront*', 'sucker*', 'contempt*', 'sucks', 'contempt*', 'sucks', 'contradic*', 'sucky',
                 'crap', 'tantrum*', 'crappy', 'teas*', 'critical', 'temper', 'critici*', 'tempers', 'cruel', 'threat*',
                 'crueler', 'tortur*', 'cruelest', 'trick', 'cruelty', 'tricked', 'cunt*', 'trickier', 'cynic*', 'trickiest',
                 'damn*', 'tricks', 'despise', 'tricky', 'destroy*', 'uglier', 'destruct', 'ugliest', 'destructed', 'ugly',
                 'vicious', 'destructive', 'viciously', 'destructivness', 'viciousness', 'distrust*', 'vile', 'distrust',
                 'villain*', 'dumb', 'violat*', 'dumbass*', 'violent', 'destruction', 'dumber', 'violently', 'dumbest',
                 'war', 'dummy', 'warfare*', 'enemie*', 'warred', 'enemy*', 'warring', 'enrag*', 'wars', 'envie*',
                 'weapon*', 'envious', 'wicked', 'envy*', 'wickedly', 'feroc*', 'yell', 'feud*', 'yelled', 'fiery', 'yelling',
                 'fight*', 'yells', 'foe*', 'fought', 'frustrat*', 'tuck', 'tucked*', 'fucker*', 'fuckface*', 'fuckh*',
                 'fuckin*', 'fucks', 'fucktard', 'fucktwat', 'fuckwad', 'fume*', 'fuming', 'furious*', 'fury', 'goddam*',
                 'greed*', 'grouch*', 'grr*', 'grudg*', 'harass*', 'hate', 'hated', 'hateful*', 'hater*', 'hates',
                 'hating', 'hatred', 'heartless*', 'hell', 'hellish', 'hostil*', 'humiliat*', 'idiot*', 'insult*', 'interrup*',
                 'intimidat*', 'jealous', 'jealousies', 'jealously','jealousy', 'jerk', 'jerked', 'jerks', 'liar*', 'lied',
                 'lies', 'ludicrous*', 'lying', 'mad', 'maddening*', 'madder', 'maddest', 'maniac*', 'meaner', 'meanest',
                 'mock', 'mocked', 'mocker*', 'mocking', 'mocks', 'moron*', 'murder*', 'nag*', 'nast*', 'obnoxious*',
                 'offence*', 'offend*', 'offense', 'offenses', 'offensive', 'outrag*', 'pest*', 'pettier', 'pettiest',
                 'petty', 'piss*', '']


        for d in data:
            z = nltk.word_tokenize(' '.join(d['text']))
            words = len(z)
            passed_dist = 0
            distance = 0
            count = 0
            for i in range(0, words):
                passed_dist += len(z[i])
                if z[i] in first_person:
                    for ang in angry:
                        distances = []
                        for j in range(0, len(d['text'])):
                            dist = find_liwc_string(d['text'][j], ang)
                            if dist > i:
                                distances.append(dist - passed_dist)
                            elif dist > 0:
                                distances.append(passed_dist - dist)
                        if len(distances) != 0:
                            distance += min(distances)
                            count += 1
            avg_dist = 1000000
            if count != 0:
                avg_dist = distance / count
            ang_fp_feat.append(avg_dist)
        ang_fp_feat = np.array(ang_fp_feat)
        extra_feats.append(ang_fp_feat)

        for d in data:
            z = nltk.word_tokenize(' '.join(d['text']))
            words = len(z)
            passed_dist = 0
            distance = 0
            count = 0
            for i in range(0, words):
                passed_dist += len(z[i])
                if z[i] in third_person:
                    for ang in angry:
                        distances = []
                        for j in range(0, len(d['text'])):
                            dist = find_liwc_string(d['text'][j], ang)
                            if dist > i:
                                distances.append(dist - passed_dist)
                            elif dist > 0:
                                distances.append(passed_dist - dist)
                        if len(distances) != 0:
                            distance += min(distances)
                            count += 1
            avg_dist = 1000000
            if count != 0:
                avg_dist = distance / count
            ang_tp_feat.append(avg_dist)

        ang_tp_feat = np.array(ang_tp_feat)
        extra_feats.append(ang_tp_feat)

    # positive emotion + pronouns
    if "pos_emo" in features:
        first_person = ['I', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']
        third_person = ['he', 'she', 'him', 'her', 'his', 'her', 'hers',
                        'they', 'them', 'their', 'theirs']
        fp_feat = []
        tp_feat = []

        pos_emo = ["accept", "accepta*", "accepted", "accepting", "accepts", "active", "actively", "admir*", "ador*",
                "advantag*", "adventur*", "affection*", "agree", "agreeable", "agreeableness", "agreeably", "agreed",
                "agreeing", "agreement*", "agrees", "alright*", "amaze*", "amazing", "amazingly", "amor*", "amus*",
                "aok", "appreciat*", "approv*", "assur*", "attract", "attracted", "attracting", "attraction",
                "attracts", "award*", "awesome", "beautiful", "beautify", "beauty", "beloved", "benefic*", "benefit",
                "benefits", "benefitt*", "benevolen*", "best", "bestest", "bestie", "besties", "better", "bless*",
                "entertain*", "enthus*", "excel", "excelled", "excellence", "excellent", "excellently", "excelling",
                "excels", "excite", "excited", "excitedly", "excitement", "exciting", "fab", "fabulous", "fabulously",
                "fabulousness", "fair", "fairer", "fairest", "faith*", "fantasi*", "fantastic", "fantastical",
                "fantastically", "fantasy", "fav", "fave", "favor", "favoring", "favorite", "favors", "favour*",
                "fearless*", "festiv*", "fiesta*", "fine", "finer", "finest", "flatter*", "flawless*", "flexib*",
                "flirt", "flirtatious", "flirting", "flirts", "flirty", "fond", "fondly", "fondness", "forgave",
                "forgiv*", "fortunately", "hugs", "humor*", "humour*", "hurra*", "ideal*", "importance", "important",
                "importantly", "impress*", "improve*", "improving", "incentive*", "innocen*", "inspir*", "intellect*",
                "intelligence", "intelligent", "interest", "interested", "interesting", "interests", "invigor*",
                "joke*", "joking", "jolly", "joy*", "keen*", "kidding", "kind", "kindly", "kindn*", "kiss*", "laidback",
                "laugh*", "legit", "libert*", "to like", "(i) like*", "(you) like*", "(we) like*", "(they) like*",
                "(do) like", "(don't) like", "(did) like", "(didn't) like", "(will) like", "(won't) like",
                "(does) like", "(doesn't) like", "(did not) like", "(will not) like", "(do not) like",
                "(does not) like", "promising", "proud", "prouder", "proudest", "proudly", "radian*", "readiness",
                "ready", "reassur*", "reinvigor*", "rejoice*", "relax*", "relief", "reliev*", "resolv*", "respect",
                "respected", "respectful", "respectfully", "respecting", "reward*", "rich", "richer", "riches",
                "richest", "rofl*", "romanc*", "romantic*", "safe", "safely", "safer", "safest", "safety", "satisf*",
                "save", "scrumptious*", "secur*", "sentimental*", "sexy", "share", "shared", "shares", "sharing",
                "sillier", "silliest", "silly", "sincer*", "smart", "smarter", "smartest", "smartly", "smil*",
                "sociability", "sociable", "wealthy", "welcom*", "well", "wellbeing", "wellness", "win", "winn*",
                "wins", "wisdom", "wise", "wisely", "wiser", "wisest", "won", "wonderful", "wonderfully", "worship*",
                "worthwhile", "wow*", "yay*", "yum", "yummy", "bliss*", "bold", "bolder", "boldest", "boldly", "bonus*",
                "brave", "braved", "braver", "bravery", "braves", "bravest", "bright", "brilliance*", "brilliant",
                "brilliantly", "calm", "calmer", "calmest", "calming", "care", "cared", "carefree", "cares", "caring",
                "certain*", "challeng*", "champ*", "charit*", "charm*", "cheer", "cheerful", "cheers", "cheery",
                "cherish*", "chuckl*", "clever", "comed*", "comfort", "comfortable", "comfortably", "comforting",
                "comforts", "compassion*", "compliment*", "confidence", "confident", "confidently", "considerate",
                "contented*", "contentment", "cool", "courag*", "create", "created", "creates", "creating", "creation",
                "creations", "free", "free-think*", "freed*", "freeing", "freely", "frees*", "freethink*", "fun",
                "funner", "funnest", "funnier", "funnies", "funniest", "funnily", "funniness", "funny", "genero*",
                "gentle", "gentler", "gentlest", "gently", "giggl*", "giver*", "giving", "glad", "gladly", "glamor*",
                "glamour*", "glori*", "glory", "good", "goodness", "gorgeous", "gorgeously", "gorgeousness", "grace",
                "graced", "graceful*", "graces", "graci*", "grand", "grande*", "gratef*", "grati*", "great", "greater",
                "greatest", "greatness", "grin", "grinn*", "grins", "ha", "hah", "haha*", "handsome", "handsomely",
                "handsomest", "happier", "happiest", "(would not) like", "(should not) like", "(could not) like",
                "(discrep) like*", "likeab*", "liked", "likes", "liking", "livel*", "lmao*", "lmfao*", "love", "loved",
                "lovelier", "loveliest", "lovely", "lover*", "loves", "loving*", "loyal", "loyalt*", "luck", "lucked",
                "luckier", "luckiest", "luckily", "lucky", "luv", "magnific*", "merit*", "merr*", "neat", "neater",
                "neatest", "neatness", "nice", "nicely", "niceness*", "nicer", "nicest", "niceties", "nurtur*", "ok",
                "okay", "okayed", "okays", "okey*", "oks", "open-minded*", "openminded*", "openness", "opportun*",
                "optimal*", "optimism", "optimistic", "original", "outgoing", "paradise*", "soulmate*", "special",
                "splendid", "splendidly", "splendor", "strength*", "strong", "stronger", "strongest", "strongly",
                "succeed*", "success", "successes", "successful", "successfully", "sunmer", "sunniest", "sunny",
                "sunshin*", "super", "superb*", "superior", "support", "supported", "supporter*", "supporting",
                "supportive", "supports", "suprem*", "sure*", "surprise", "surprised*", "surprising*", "sweet",
                "sweeter", "sweetest", "sweetheart*", "sweetie*", "sweetly", "sweetness*", "sweets", "talent*",
                "teehe*", "tender", "tenderly", "terrific", "terrifically", "thank", "thanked", "thankful",
                "thankfully", "thanking", "thanks", "thanx", "thnx", "thoughtful*", "thrill*", "thx", "toleran*",
                "credit*", "cute", "cuter", "cutest", "cutie*", "daring", "darlin*", "dear", "dearly", "decent",
                "definitely", "delectabl *", "delicate*", "delicious*", "deligh*", "desir*", "determina*", "determined",
                "devot*", "dignified", "dignifies", "dignifying", "dignity", "divin*", "eager", "eagerly", "eagerness",
                "ease*", "easier", "easiest", "easily", "easiness", "easing", "easy*", "ecsta*", "elegan*", "encourag*",
                "energ*", "engag*", "enjoy*", "happy", "harmon*", "heal", "healed", "healer*", "healing", "heals",
                "healthy", "heartfelt", "heartwarm*", "heaven*", "helper*", "helpful", "helpfully", "helpfulness",
                "helping", "helps", "hero", "hero's", "heroes", "heroic*", "heroine*", "heroism", "hilarious", "hoho*",
                "honest", "honestly", "honesty", "honor*", "honour*", "hooray", "hope", "hoped", "hopeful", "hopefully",
                "hopes", "hoping", "hug", "hugg*", "passion*", "peace", "peaceful", "peacefully", "peacekeep*",
                "peacemak*", "perfect", "perfected", "perfecting", "perfection", "perfectly", "play", "played",
                "playful", "playfully", "playfulness", "playing", "plays", "pleasant*", "please*", "pleasing",
                "pleasur*", "polite", "politely", "popular", "populari*", "positive", "positively", "positives",
                "positivi*", "prais*", "precious*", "prettier", "prettiest", "pretty", "pride", "privileg*", "prize*",
                "profit*", "promise*", "treat", "triumph*", "true", "truer", "truest", "truly", "trust", "trusted",
                "trusting", "trusts", "trustworthiness", "trustworthy", "trusty", "truth*", "ty", "upbeat", "useful",
                "usefully", "usefulness", "valuabl*", "value", "valued", "values", "valuing", "vigor*", "vigour*",
                "virtue*", "virtuo*", "vital*", "warm", "warmed", "warmer", "warmest", "warming", "warmly", "warms",
                "warmth", "wealth", "wealthier", "wealthiest"]
        for d in data:
            z = nltk.word_tokenize(' '.join(d['text']))
            words = len(z)
            passed_dist = 0
            distance = 0
            count = 0
            for i in range(0, words):
                passed_dist += len(z[i])
                if z[i] in first_person:
                    for sad in pos_emo:
                        distances = []
                        for j in range(0, len(d['text'])):
                            dist = find_liwc_string(d['text'][j], sad)
                            if dist > passed_dist:
                                distances.append(dist - passed_dist)
                            elif dist > 0:
                                distances.append(passed_dist - dist)
                        if len(distances) != 0:
                            distance += min(distances)
                            count += 1
            avg_dist = 1000000
            if count != 0:
                avg_dist = distance / count
            fp_feat.append(avg_dist)
        fp_feat = np.array(fp_feat)
        extra_feats.append(fp_feat)

        for d in data:
            z = nltk.word_tokenize(' '.join(d['text']))
            words = len(z)
            passed_dist = 0
            distance = 0
            count = 0
            for i in range(0, words):
                passed_dist += len(z[i])
                if z[i] in third_person:
                    for sad in pos_emo:
                        distances = []
                        for j in range(0, len(d['text'])):
                            dist = find_liwc_string(d['text'][j], sad)
                            if dist > passed_dist:
                                distances.append(dist - passed_dist)
                            elif dist > 0:
                                distances.append(passed_dist - dist)
                        if len(distances) != 0:
                            distance += min(distances)
                            count += 1
            avg_dist = 1000000
            if count != 0:
                avg_dist = distance / count
            tp_feat.append(avg_dist)

        tp_feat = np.array(tp_feat)
        extra_feats.append(tp_feat)

    # pronouns
    if "pronouns" in features:
        first_person = ['I', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']
        third_person = ['he', 'she', 'him', 'her', 'his', 'her', 'hers',
                        'they', 'them', 'their', 'theirs']
        fp_feat = []
        tp_feat = []

        emotion_words = ['stressed', 'upset', 'angry', 'hopeless', 'ashamed', 'weak', 'terrified', 'scared', 'lost',
                         'anxious', 'overwhelmed', 'panic', 'panicking', 'anxiety', 'stress', "panic*",
                         "afraid", "alarm*", "anxiety", "anxious", "anxiously", "anxiousness", "apprehens*", "asham*",
                         "aversi*", "avoid*", "awkward", "confuse", "confused", "confusedly", "confusing", "desperat*",
                         "discomfort*", "distraught", "distress*", "disturb*", "doubt*", "dread*", "dwell*",
                         "embarrass*", "fear", "feared", "fearful*", "fearing", "fears", "frantic*", "fright*", "guilt",
                         "guilt-trip", "guilty", "hesita*", "horrible", "horriby", "horrid*", "horror*", "humiliat*",
                         "impatien*", "inadequa*", "indecis*", "inhibit*", "insecur*", "irrational*", "irrita*",
                         "miser*", "nervous", "nervously", "nervousness", "neurotic*", "obsess*", "overwhelm*",
                         "paranoi*", "petrif*", "phobi*", "pressur*", "reluctan*", "repress*", "restless*", "rigid",
                         "rigidity", "rigidly", "risk*", "scare", "scared", "scares", "scarier", "scariest", "scaring",
                         "scary", "shake*", "shaki*", "shaky", "shame*", "shook", "shy", "shyly", "shyness", "startl*",
                         "strain*", "stress*", "struggl*", "suspicio*", "tense", "tensely", "tensing", "tension*",
                         "terrified", "terrifies", "terrify", "terrifying", "terror*", "threat*", "timid*", "trembl*",
                         "turmoil", "twitchy", "uncertain*", "uncomfortabl*", "uncontrol*", "uneas*", "unsettl*",
                         "unsure*", "upset", "upsets", "upsetting", "uptight*", "vulnerab*", "worried", "worrier",
                         "worries", "worry", "worrying", "abandon*", "abuse*", "abusi*", "ache*", "aching*", "advers*",
                         "afraid", "aggravat*", "aggress", "aggressed", "aggresses", "aggressing", "aggression*",
                         "aggressive", "aggressively", "aggressor*", "agitat*", "agoniz*", "agony", "alarm*", "alone",
                         "anger*", "angrier", "angriest", "angry", "anguish*", "annoy", "annoyed", "annoying", "annoys",
                         "antagoni*", "anxiety", "anxious", "anxiously", "anxiousness", "apath*", "appall*",
                         "apprehens*", "argh*", "argu*", "arrogan*", "asham*", "assault*", "attack*", "avers*",
                         "avoid*", "awful", "awkward", "bad", "badly", "bashful*", "bastard*", "battl*", "beaten",
                         "bereave*", "bitch*", "disliking", "dismay*", "disreput*", "diss", "dissatisf*", "distraught",
                         "distress*", "distrust*", "disturb*", "domina*", "doom*", "dork*", "doubt*", "dread*", "dull",
                         "dumb", "dumbass*", "dumber", "dumbest", "dummy", "dump*", "dwell*", "egotis*", "embarrass*",
                         "emotional", "emptier", "emptiest", "emptiness", "empty", "enemie*", "enemy*", "enrag*",
                         "envie*", "envious", "envy*", "evil", "excruciat*", "exhaust*", "fail*", "fake", "fatal*",
                         "fatigu*", "fault*", "fear", "feared", "fearful*", "fearing", "fears", "feroc*", "feud*",
                         "fiery", "fight*", "fired", "flunk*", "fool", "fooled", "fooling", "foolish", "ignoramus",
                         "ignorant", "ignore", "ignored", "ignores", "ignoring", "immoral*", "impatien*", "impersonal",
                         "impolite*", "inadequa*", "incompeten*", "indecis*", "ineffect*", "inferior", "inferiority",
                         "inhibit*", "insecur*", "insincer*", "insult*", "interrup*", "intimidat*", "irrational*",
                         "irrita*", "isolat*", "jaded", "jealous", "jealousies", "jealously", "jealousy", "jerk",
                         "jerked", "jerks", "kill*", "lame", "lamely", "lameness", "lamer", "lamest", "lazier",
                         "laziest", "lazy", "liabilit*", "liar*", "lied", "lies", "lone", "lonelier", "loneliest",
                         "loneliness", "lonely", "loner*", "longing*", "lose", "loser*", "loses", "losing", "lost",
                         "poorer", "poorest", "poorly", "poorness*", "powerless*", "prejudic*", "pressur*", "prick*",
                         "problem*", "protest", "protested", "protesting", "protests", "puk*", "punish*", "pushy",
                         "queas*", "rage*", "raging", "rancid*", "rape*", "raping", "rapist*", "rebel*", "reek*",
                         "regret*", "reject*", "reluctan*", "remorse*", "repress*", "resent*", "resign*", "restless*",
                         "revenge*", "ridicul*", "rigid", "rigidity", "rigidly", "risk*", "rotten", "rude", "rudely",
                         "rum*", "sad", "sadder", "saddest", "sadly", "sadness", "sarcas*", "savage*", "scare",
                         "scared", "scares", "scarier", "scariest", "scaring", "scary", "sceptic*", "scream*", "tragic",
                         "tragically", "trauma*", "trembl*", "trick", "tricked", "trickier", "trickiest", "tricks",
                         "tricky", "trite", "trivial", "troubl*", "turmoil", "twitchy", "ugh", "uglier", "ugliest",
                         "ugly", "unaccept*", "unattractive", "uncertain*", "uncomfortabl*", "uncontrol*", "undesir*",
                         "uneas*", "unfair", "unfortunate*", "unfriendly", "ungrateful*", "unhapp*", "unimportant",
                         "unimpress*", "unkind", "unlov*", "unlucky", "unpleasant", "unprotected", "unsafe", "unsavory",
                         "unsettl*", "unsuccessful*", "unsure*", "unwelcom*", "upset", "upsets", "upsetting",
                         "uptight*", "useless", "uselessly", "uselessness", "vain", "vanity", "vicious", "viciously",
                         "viciousness", "victim*", "vile", "villain*", "bitter", "bitterly", "bitterness", "blam*",
                         "bore*", "boring", "bother*", "broke", "brutal*", "burden*", "careless*", "cheat*", "coldly",
                         "complain*", "condemn*", "confront*", "confuse", "confused", "confusedly", "confusing",
                         "contempt*", "contradic*", "crap", "crappy", "crazy", "cried", "cries", "critical", "critici*",
                         "crude", "crudely", "cruel", "crueler", "cruelest", "cruelty", "crushed", "cry", "crying",
                         "cunt*", "curse", "cut", "cynic*", "damag*", "damn*", "danger", "dangerous", "dangerously",
                         "dangers", "daze*", "decay*", "deceptive", "deceiv*", "defeat*", "defect*", "defenc*",
                         "defend*", "defense", "defenseless", "defensive", "defensively", "defensiveness", "foolishly",
                         "fools", "forbade", "forbid", "forbidden", "forbidding", "forbids", "fought", "frantic*",
                         "freak*", "fright*", "frustrat*", "fuck", "fucked*", "fucker*", "fuckface*", "fuckh*",
                         "fuckin*", "fucks", "fucktard", "fucktwat*", "fuckwad*", "fume*", "fuming", "furious*", "fury",
                         "geek*", "gloom", "gloomier", "gloomiest", "gloomily", "gloominess", "gloomy", "goddam*",
                         "good-for-nothing", "gossip*", "grave*", "greed*", "grief", "griev*", "grim", "grimac*",
                         "grimly", "gross", "grossed", "grosser", "grossest", "grossing", "grossly", "grossness",
                         "grouch*", "grr*", "grudg*", "guilt", "guilt-trip*", "guiltier", "guiltiest", "guilty",
                         "hangover*", "harass*", "harm", "lous*", "loveless", "low", "lower", "lowered", "lowering",
                         "lowers", "lowest", "lowli*", "lowly", "luckless*", "ludicrous*", "lying", "mad", "maddening*",
                         "madder", "maddest", "maniac*", "masochis*", "meaner", "meanest", "melanchol*", "mess",
                         "messier", "messiest", "messy", "miser*", "miss", "missed", "misses", "missing", "mistak*",
                         "mock", "mocked", "mocker*", "mocking", "mocks", "molest*", "mooch*", "moodi*", "moody",
                         "moron*", "mourn*", "murder*", "nag*", "nast*", "needy", "neglect*", "nerd*", "nervous",
                         "nervously", "nervousness", "neurotic*", "nightmar*", "numbed", "numbing", "numbness", "numb*",
                         "obnoxious*", "obsess*", "offence*", "screw*", "selfish*", "serious", "seriously",
                         "seriousness", "severe*", "shake*", "shaki*", "shaky", "shame*", "shit*", "shock*", "shook",
                         "shy", "shyly", "shyness", "sick", "sicken*", "sicker", "sickest", "sickly", "sigh", "sighed",
                         "sighing", "sighs", "sin", "sinister", "sins", "slut*", "smh", "smother*", "smug*", "snob*",
                         "sob", "sobbed", "sobbing", "sobs", "solemn*", "sorrow*", "sorry", "spite*", "stale",
                         "stammer*", "stank*", "startl*", "steal*", "stench*", "stink", "stinky", "strain*", "strange",
                         "strangest", "stress*", "struggl*", "stubborn*", "stunk", "stupid", "stupider", "stupidest",
                         "stupidity", "stupidly", "violat*", "violence", "violent", "violently", "vomit*", "vulnerab*",
                         "war", "warfare*", "warn*", "warred", "warring", "wars", "weak", "weaken", "weakened",
                         "weakening", "weakens", "weaker", "weakest", "weakling", "weakly", "weapon*", "weary", "weep*",
                         "weird", "weirded", "weirder", "weirdest", "weirdly", "weirdness", "weirdo", "weirdos",
                         "weirds", "wept", "whine*", "whining", "whore*", "wicked", "wickedly", "wimp*", "witch*",
                         "woe*", "worried", "worrier", "worries", "worry", "worrying", "worse", "worsen", "worsened",
                         "worsening", "worsens", "worst", "worthless", "wrong", "wrongdoing", "wronged", "wrongful",
                         "wrongly", "wrongness", "wrongs", "degrad*", "demean*", "demot*", "denial", "depress*",
                         "depriv*", "despair*", "desperat*", "despis*", "destroy*", "destruct", "destructed",
                         "destruction", "destructive", "destructivness", "devastat*", "devensiveness", "devil*",
                         "difficult", "difficulties", "difficulty", "disadvantag*", "disagree*", "disappoint*",
                         "disaster*", "discomfort*", "discourag*", "disgrac*", "disgust*", "dishearten*", "dishonor*",
                         "disillusion*", "dislike", "disliked", "dislikes", "harmed", "harmful", "harmfully",
                         "harmfulness", "harming", "harms", "harsh", "hate", "hated", "hateful*", "hater*", "hates",
                         "hating", "hatred", "haunted", "hazard*", "heartbreak*", "heartbroke*", "heartless*", "hell",
                         "hellish", "helpless*", "hesita*", "homesick*", "hopeless*", "horrible", "horribly", "horrid*",
                         "horror*", "hostil*", "humiliat*", "hungover", "hurt*", "idiot*", "ignorable", "offend*",
                         "offense", "offenses", "offensive", "outrag*", "overwhelm*", "pain", "pained", "painf*",
                         "pains", "pamc*", "paranm*", "pathetic", "pathetically", "peculiar*", "perv", "perver*",
                         "pervy", "pessims*", "pest*", "petrif*", "pettier", "pettiest", "petty", "phobi*", "phony",
                         "piss*", "pitiable", "pitied", "pities", "pitiful", "pitifully", "pity*", "poison*", "poor",
                         "stutter*", "suck", "sucked", "sucker*", "sucks", "sucky", "suffer", "suffered", "sufferer*",
                         "suffering", "suffers", "suspicio*", "tantrum*", "tears", "teas*", "tedious", "temper",
                         "tempers", "tense", "tensely", "tensing", "tension*", "terrible", "terribly", "terrified",
                         "terrifies", "terrify", "terrifying", "terror*", "thief", "thiev*", "threat*", "timid*",
                         "tortur*", "traged*", "yearn*", "yell", "yelled", "yelling", "yells", "yuck"]
        for d in data:
            z = nltk.word_tokenize(' '.join(d['text']))
            words = len(z)
            passed_dist = 0
            distance = 0
            count = 0
            for i in range(0, words):
                passed_dist += len(z[i])
                if z[i] in first_person:
                    for sad in emotion_words:
                        distances = []
                        for j in range(0, len(d['text'])):
                            dist = find_liwc_string(d['text'][j], sad)
                            if dist > i:
                                distances.append(dist - passed_dist)
                            elif dist > 0:
                                distances.append(passed_dist - dist)
                        if len(distances) != 0:
                            distance += min(distances)
                            count += 1
            avg_dist = 1000000
            if count != 0:
                avg_dist = distance / count
            fp_feat.append(avg_dist)
        fp_feat = np.array(fp_feat)
        extra_feats.append(fp_feat)

        for d in data:
            z = nltk.word_tokenize(' '.join(d['text']))
            words = len(z)
            passed_dist = 0
            distance = 0
            count = 0
            for i in range(0, words):
                passed_dist += len(z[i])
                if z[i] in third_person:
                    for sad in emotion_words:
                        distances = []
                        for j in range(0, len(d['text'])):
                            dist = find_liwc_string(d['text'][j], sad)
                            if dist > i:
                                distances.append(dist - passed_dist)
                            elif dist > 0:
                                distances.append(passed_dist - dist)
                        if len(distances) != 0:
                            distance += min(distances)
                            count += 1
            avg_dist = 1000000
            if count != 0:
                avg_dist = distance / count
            tp_feat.append(avg_dist)

        tp_feat = np.array(tp_feat)
        extra_feats.append(tp_feat)


    # Diana's attempt
    # if "dianas_feat" in features:
    #     stress_words = ['anxious', 'bad', 'ashamed', 'stressed', 'sick', 'enraged', 'weak', 'terrified', 'shitty',
    #                     'lost',
    #                     'alone', 'anxious', 'worse', 'crappy', 'sad', 'lonely', 'upset', 'sad', 'afraid', 'scared']
    #     my_feat = []
    #     found = False
    #     for d in data:
    #         z = [nltk.word_tokenize(' '.join(d['text']))]
    #         i = 0
    #         for word in z:
    #             if word == 'I' and z[i + 1] == 'feel':
    #                 for stress in stress_words:
    #                     if z[i + 2] == stress:
    #                         my_feat.append(1)
    #                         found = True
    #             if not found:
    #                 my_feat.append(0)
    #             i += 1
    #     my_feat = np.array(my_feat)
    #     extra_feats.append(my_feat)

    # add on the extra features, scale things into the same space, and return the data
    if len(extra_feats) > 0:
        extra_feats = np.vstack(extra_feats).transpose()
        if sparse.issparse(X):  # the n-gram vectorizer outputs a sparse matrix; the word embeddings do not
            # we want to scale the extra features so they are all in the same space, but can't scale ngram counts
            extra_feats = scaler.transform(extra_feats)
            X = np.hstack((X.todense(), extra_feats))
            if args.sparse:
                X = sparse.csr_matrix(X)
        else:
            # but we can scale word embeddings! hopefully this is a good idea
            X = np.hstack((X, extra_feats))
            X = scaler.transform(X)
    elif sparse.issparse(X) and not args.sparse:
        X = X.todense()

    if labels:
        y = [d['label'] for d in data]
        return X, y
    else:
        return X


########################################################################################################################
# Grid search over a given parameter space with the given data (using 5-fold cross-validation);
# also return the best classifier.
# X : the observations
# y : the labels
# classifier : an instance of the type of classifier (e.g., a plain SVM or Naive Bayes classifier)
# grid : a dictionary specifying the parameter space to search
def grid_search(X, y, classifier, grid):
    print("Fitting model", classifier, "...")

    model = classifier()
    # stratified K-fold maintains the class proportions in each split
    # we also reset the random seed so we are always doing the same cross-validation
    cv_splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    clf = GridSearchCV(model, grid, cv=cv_splitter)
    clf.fit(X, y)

    return clf.best_score_, clf.best_estimator_


########################################################################################################################
# Test a lot of classifiers by cross-validation on the train set; report the result of the best one on the test set.
def train(main_args):
    try:
        main_args.param_grid
    except:
        main_args.param_grid = {
            svm.SVC: [
                {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
                {'C': [1, 10, 100, 1000], 'kernel': ['poly'], 'gamma': [0.001, 0.0001], 'degree': [2, 3, 4]},
                {'C': [1, 10, 100, 1000], 'kernel': ['poly'], 'degree': [2, 3, 4]},
                {'C': [1, 10, 100, 1000], 'kernel': ['rbf', 'sigmoid'], 'gamma': [0.001, 0.0001]},
                {'C': [1, 10, 100, 1000], 'kernel': ['rbf', 'sigmoid']}
            ], linear_model.LogisticRegression: [
                {'penalty': ['l1', 'l2'], "C": [0.5, 1, 10, 50, 100, 500, 1000], 'max_iter': [200]}
            ], naive_bayes.GaussianNB: [{}]
        }

    best_cv = 0.0
    clf = dummy.DummyClassifier(strategy='most_frequent')
    best_scale = None
    best_vec = None
    best_args = None

    if main_args.embedding_mode == 'ngram':
        # NGRAM CV LOOP
        for vocab in [5000, 10000, 20000]:
            for nmin, nmax in [(1, 1), (1, 2), (1, 3)]:
                print("Trying vocab size", vocab, "with ngrams from", nmin, "to", nmax)

                # Gaussian Naive Bayes has to be a pain and refuse to use sparse data
                if naive_bayes.GaussianNB in main_args.param_grid:
                    args = argparse.Namespace(min_confidence=main_args.confidence, embedding_mode="ngram", max_df=1.0,
                                              max_features=vocab, min_ngram=nmin, max_ngram=nmax, sparse=False)
                    X, y, scaler, vectorizer = establish_preprocessing(LABELED_DATA_FN, args, main_args.features, True)

                    score, fitted_clf = grid_search(X, y, naive_bayes.GaussianNB, [{}])
                    if score > best_cv:
                        best_cv = score
                        clf = fitted_clfr
                        best_scale = scaler
                        best_vec = vectorizer
                        best_args = args

                # now try everything that isn't Gaussian NB
                args = argparse.Namespace(min_confidence=main_args.confidence, embedding_mode="ngram", max_df=1.0,
                                          max_features=vocab, min_ngram=nmin, max_ngram=nmax, sparse=True)
                X, y, scaler, vectorizer = establish_preprocessing(LABELED_DATA_FN, args, main_args.features, True)

                for candidate_clf in main_args.param_grid.keys():
                    if candidate_clf == naive_bayes.GaussianNB:
                        continue
                    score, fitted_clf = grid_search(X, y, candidate_clf, main_args.param_grid[candidate_clf])
                    if score > best_cv:
                        best_cv = score
                        clf = fitted_clf
                        best_scale = scaler
                        best_vec = vectorizer
                        best_args = args
    else:
        # WORD2VEC CV LOOP
        args = argparse.Namespace(min_confidence=main_args.confidence, embedding_mode=main_args.embedding_mode,
                                  sparse=False)
        X, y, scaler = establish_preprocessing(LABELED_DATA_FN, args, main_args.features, True)

        for candidate_clf in main_args.param_grid.keys():
            score, fitted_clf = grid_search(X, y, candidate_clf, main_args.param_grid[candidate_clf])
            if score > best_cv:
                best_cv = score
                clf = fitted_clf
                best_scale = scaler
                best_args = args

    print("Results of cross-validation for confidence level", main_args.confidence, ":", best_cv, "from classifier",
          clf, "\nwith args", best_args, "\nand features", main_args.features, "\n\n")

    # now run on the test set to produce F-score metrics
    with open(TEST_DATA_FN, "rb") as f:
        test = pickle.load(f)

    X_test, y_test = preprocess_data(test, best_scale, best_args, main_args.features, vectorizer=best_vec, labels=True)

    y_pred = clf.predict(X_test)

    print("Precision, recall, F-score, support:")
    print(prf_score(y_test, y_pred, average='binary'))

    # #################################################################
    # # attempt to test significance
    # #################################################################
    # contingency_table = contingency_matrix(y_test, y_pred)
    # print("contingency matrix for " + clf + "\n" + contingency_table)
    # # print(mcnemar(contingency_table))
    #
    # # return the classifier and its preprocessing information (args, vectorizer/scaler)
    # # in case we want to do anything with it, like label extra data or compare it to another classifier
    # return clf, best_scale, best_args, best_vec


########################################################################################################################
# Main. Test models and report results.
if __name__ == "__main__":
    embedding_mode = "word2vec"
    print("EMBEDDING MODE IS", embedding_mode)

    # TEST: all features + pronouns
    all_features_p = ['liwc_i', 'liwc_negemo','pronouns']

    # test all features together
    print("liwc i and negemo + pronouns")
    train(argparse.Namespace(features=all_features_p, confidence=0.8, embedding_mode=embedding_mode))  # all features on 4/5 data

    # print("now with pronouns")
    # train(argparse.Namespace(features=['pronouns'], confidence=0, embedding_mode=embedding_mode))
    # train(argparse.Namespace(features=['pronouns'], confidence=0.6, embedding_mode=embedding_mode))
    # train(argparse.Namespace(features=['pronouns'], confidence=0.8, embedding_mode=embedding_mode))

    # TEST: baseline - no features
    # train(argparse.Namespace(features=[], confidence=0, embedding_mode=embedding_mode))
    # train(argparse.Namespace(features=[], confidence=0.6, embedding_mode=embedding_mode))
    # train(argparse.Namespace(features=[], confidence=0.8, embedding_mode=embedding_mode))

    # print("now with fear pronouns")
    # train(argparse.Namespace(features=['fear'], confidence=0, embedding_mode=embedding_mode))
    # train(argparse.Namespace(features=['fear'], confidence=0.6, embedding_mode=embedding_mode))
    # train(argparse.Namespace(features=['fear'], confidence=0.8, embedding_mode=embedding_mode))
    #
    # print("now with anger pronouns")
    # train(argparse.Namespace(features=['anger'], confidence=0, embedding_mode=embedding_mode))
    # train(argparse.Namespace(features=['anger'], confidence=0.6, embedding_mode=embedding_mode))
    # train(argparse.Namespace(features=['anger'], confidence=0.8, embedding_mode=embedding_mode))

    # print("now with diana's feat")
    # train(argparse.Namespace(features=['dianas_feat'], confidence=0, embedding_mode=embedding_mode))
    # train(argparse.Namespace(features=['dianas_feat'], confidence=0.6, embedding_mode=embedding_mode))
    # train(argparse.Namespace(features=['dianas_feat'], confidence=0.8, embedding_mode=embedding_mode))

    # # TEST: all features + pronouns
    # all_features_p = ['timestamps', 'sentiment', 'dal_max_pleasantness', 'dal_max_activation', 'dal_max_imagery',
    #                   'dal_min_pleasantness', 'dal_min_activation', 'dal_min_imagery', 'liwc_WC', 'liwc_Analytic',
    #                   'liwc_Clout', 'liwc_Authentic', 'liwc_Tone', 'liwc_WPS', 'liwc_Sixltr', 'liwc_Dic', 'liwc_function',
    #                   'liwc_pronoun', 'liwc_ppron', 'liwc_i', 'liwc_we', 'liwc_you', 'liwc_shehe', 'liwc_they',
    #                   'liwc_ipron', 'liwc_article', 'liwc_prep', 'liwc_auxverb', 'liwc_adverb', 'liwc_conj',
    #                   'liwc_negate', 'liwc_verb', 'liwc_adj', 'liwc_compare', 'liwc_interrog', 'liwc_number',
    #                   'liwc_quant', 'liwc_affect', 'liwc_posemo', 'liwc_negemo', 'liwc_anx', 'liwc_anger', 'liwc_sad',
    #                   'liwc_social', 'liwc_family', 'liwc_friend', 'liwc_female', 'liwc_male', 'liwc_cogproc',
    #                   'liwc_insight', 'liwc_cause', 'liwc_discrep', 'liwc_tentat', 'liwc_certain', 'liwc_differ',
    #                   'liwc_percept', 'liwc_see', 'liwc_hear', 'liwc_feel', 'liwc_bio', 'liwc_body', 'liwc_health',
    #                   'liwc_sexual', 'liwc_ingest', 'liwc_drives', 'liwc_affiliation', 'liwc_achieve', 'liwc_power',
    #                   'liwc_reward', 'liwc_risk', 'liwc_focuspast', 'liwc_focuspresent', 'liwc_focusfuture',
    #                   'liwc_relativ', 'liwc_motion', 'liwc_space', 'liwc_time', 'liwc_work', 'liwc_leisure', 'liwc_home',
    #                   'liwc_money', 'liwc_relig', 'liwc_death', 'liwc_informal', 'liwc_swear', 'liwc_netspeak',
    #                   'liwc_assent', 'liwc_nonflu', 'liwc_filler', 'liwc_AllPunc', 'liwc_Period', 'liwc_Comma',
    #                   'liwc_Colon', 'liwc_SemiC', 'liwc_QMark', 'liwc_Exclam', 'liwc_Dash', 'liwc_Quote', 'liwc_Apostro',
    #                   'liwc_Parenth', 'liwc_OtherP', 'dal_avg_activation', 'dal_avg_imagery', 'dal_avg_pleasantness',
    #                   'pronouns']
    #
    # # test all features together
    # print("all features + pronouns")
    # train(argparse.Namespace(features=all_features_p, confidence=0, embedding_mode=embedding_mode))  # all features on all data
    # train(argparse.Namespace(features=all_features_p, confidence=0.6, embedding_mode=embedding_mode))  # all features on 3/5 data
    # train(argparse.Namespace(features=all_features_p, confidence=0.8, embedding_mode=embedding_mode))  # all features on 4/5 data

    # # TEST: all features - pronouns
    # all_features = ['timestamps', 'sentiment', 'dal_max_pleasantness', 'dal_max_activation', 'dal_max_imagery',
    #                 'dal_min_pleasantness', 'dal_min_activation', 'dal_min_imagery', 'liwc_WC', 'liwc_Analytic',
    #                 'liwc_Clout', 'liwc_Authentic', 'liwc_Tone', 'liwc_WPS', 'liwc_Sixltr', 'liwc_Dic', 'liwc_function',
    #                 'liwc_pronoun', 'liwc_ppron', 'liwc_i', 'liwc_we', 'liwc_you', 'liwc_shehe', 'liwc_they',
    #                 'liwc_ipron', 'liwc_article', 'liwc_prep', 'liwc_auxverb', 'liwc_adverb', 'liwc_conj',
    #                 'liwc_negate', 'liwc_verb', 'liwc_adj', 'liwc_compare', 'liwc_interrog', 'liwc_number',
    #                 'liwc_quant', 'liwc_affect', 'liwc_posemo', 'liwc_negemo', 'liwc_anx', 'liwc_anger', 'liwc_sad',
    #                 'liwc_social', 'liwc_family', 'liwc_friend', 'liwc_female', 'liwc_male', 'liwc_cogproc',
    #                 'liwc_insight', 'liwc_cause', 'liwc_discrep', 'liwc_tentat', 'liwc_certain', 'liwc_differ',
    #                 'liwc_percept', 'liwc_see', 'liwc_hear', 'liwc_feel', 'liwc_bio', 'liwc_body', 'liwc_health',
    #                 'liwc_sexual', 'liwc_ingest', 'liwc_drives', 'liwc_affiliation', 'liwc_achieve', 'liwc_power',
    #                 'liwc_reward', 'liwc_risk', 'liwc_focuspast', 'liwc_focuspresent', 'liwc_focusfuture',
    #                 'liwc_relativ', 'liwc_motion', 'liwc_space', 'liwc_time', 'liwc_work', 'liwc_leisure', 'liwc_home',
    #                 'liwc_money', 'liwc_relig', 'liwc_death', 'liwc_informal', 'liwc_swear', 'liwc_netspeak',
    #                 'liwc_assent', 'liwc_nonflu', 'liwc_filler', 'liwc_AllPunc', 'liwc_Period', 'liwc_Comma',
    #                 'liwc_Colon', 'liwc_SemiC', 'liwc_QMark', 'liwc_Exclam', 'liwc_Dash', 'liwc_Quote', 'liwc_Apostro',
    #                 'liwc_Parenth', 'liwc_OtherP', 'dal_avg_activation', 'dal_avg_imagery', 'dal_avg_pleasantness']
    # print("all features w/o pronouns")
    # # train(argparse.Namespace(features=all_features, confidence=0, embedding_mode=embedding_mode))  # all features on all data
    # # train(argparse.Namespace(features=all_features, confidence=0.6, embedding_mode=embedding_mode))  # all features on 3/5 data
    # train(argparse.Namespace(features=all_features, confidence=0.8, embedding_mode=embedding_mode))  # all features on 4/5 data

    #
    #
    # # TEST: each feature type individually
    # feature_types = [['sentiment'], ['timestamps'], list(filter(lambda x: 'dal_' in x, all_features)),
    #                  list(filter(lambda x: 'liwc_' in x, all_features))]
    #
    # for feats in feature_types:
    #     train(argparse.Namespace(features=feats, confidence=0, embedding_mode=embedding_mode))
    #     train(argparse.Namespace(features=feats, confidence=0.6, embedding_mode=embedding_mode))
    #     train(argparse.Namespace(features=feats, confidence=0.8, embedding_mode=embedding_mode))
    #
    #
    # # TEST: just features with >= 0.2 absolute correlation
    # high_corr_feats = ['dal_min_pleasantness', 'sentiment', 'liwc_Clout', 'liwc_Authentic', 'liwc_Tone', 'liwc_i',
    #                    'liwc_posemo', 'liwc_negemo', 'liwc_anx', 'liwc_social']
    #
    # train(argparse.Namespace(features=high_corr_feats, confidence=0, embedding_mode=embedding_mode))
    # train(argparse.Namespace(features=high_corr_feats, confidence=0.6, embedding_mode=embedding_mode))
    # train(argparse.Namespace(features=high_corr_feats, confidence=0.8, embedding_mode=embedding_mode))
