"""Deterministic development routing; never evaluates a command string."""
import re

KINDS = ('application', 'website', 'software', 'system', 'api', 'game',
         'mobile', 'desktop', 'ai', 'automation')


def parse_intent(text):
    if not isinstance(text, str) or not 1 <= len(text.strip()) <= 4000:
        raise ValueError('Enter a development request of 1–4000 characters')
    original = text.strip()
    match = re.fullmatch(r'(?:s\s+)?(?:develop|build|create)\s+(?:(?:me\s+)?(?:an?\s+)?)?(.+)',
                         original, re.I)
    if not match:
        raise ValueError('Use S develop application, website, api, or another project type')
    description = match.group(1)
    tokens = set(re.findall(r'[a-z]+', description.lower()))
    aliases = [('website', {'website', 'webpage', 'site'}),
               ('mobile', {'mobile', 'android', 'ios'}), ('desktop', {'desktop'}),
               ('api', {'api'}), ('game', {'game'}), ('ai', {'ai'}),
               ('automation', {'automation'}), ('system', {'system'}),
               ('software', {'software'}), ('application', {'application', 'app'})]
    kind = next((kind for kind, words in aliases if words & tokens), None)
    if kind is None:
        raise ValueError('Project type unclear; specify application, website, system, api, game, mobile, desktop, ai, or automation')
    return {'kind': kind, 'request': original, 'description': description,
            'classifier': 'deterministic-keywords-v1'}
