"""Expand the existing 12-scene pilot with 12 independently authored scenes.

No network, pretrained model, GPU, or changes to existing pilot/checkpoints.
Run: python -m training.v12.expand_pilot
"""
import argparse
import json
from pathlib import Path

NEW_SCENES = [
('The Floodgate', 'adventure', ['Jana','Omar'], 'A riverside floodgate at night', 'Open the jammed floodgate before the village floods', 'A fallen beam blocks the turning wheel', 'Jana and Omar use a chain to shift the beam and release the water', '''Jana felt the floodgate shudder beneath her boots. On the far bank, lanterns bobbed as villagers climbed toward higher ground. Omar tried the iron wheel, but it would not turn. A fallen beam had wedged itself between the spokes. Jana fastened a chain around the beam while Omar hooked its other end to the maintenance winch. They pulled together. The beam shifted an inch, then broke free with a sharp crack. Omar spun the wheel. Black water rushed through the opening, and the river level began to fall below the painted danger mark.'''),
('The Borrowed Shadow', 'fantasy', ['Edda','Fen'], 'A market where shadows can be traded', 'Recover Fen’s stolen shadow before sundown', 'The shadow merchant demands a secret instead of money', 'Edda trades a harmless childhood secret and returns Fen’s shadow', '''Fen stood beneath the market awning, but the afternoon sun cast nothing at his feet. Edda followed the trail of dark footprints to a stall hung with bottled silhouettes. The merchant lifted a jar containing Fen's restless shadow. "A secret," she said, ignoring Edda's coins. Edda considered the dangerous things she knew, then offered a smaller truth: as a child, she had hidden her brother's birthday gift and forgotten where. The merchant laughed, unscrewed the jar, and let the shadow slide across the stones. It joined Fen just as the sun touched the rooftops.'''),
('Orbit of Ash', 'sci_fi', ['Kei','Luma'], 'A damaged orbital greenhouse', 'Save the last viable seedlings before the station loses pressure', 'The airlock seals cannot close around the plant trays', 'They transfer the seedlings into emergency water bags and seal the airlock', '''A thin white mist escaped through the greenhouse hatch. Kei checked the pressure gauge: six minutes before the room became uninhabitable. Luma dragged the seedling trays toward the airlock, but their metal frames jammed the door. "Leave the frames," Kei said. They eased each root ball into an emergency water bag, sealing the stems above the liquid. The bags fitted into the narrow cargo compartment. Luma slammed the inner hatch, and the pressure warning faded from red to amber. Behind the glass, the empty greenhouse went dark, but the seedlings floated safely beside them.'''),
('The Orchard Witness', 'mystery', ['Bea','Hollis'], 'An apple orchard after a theft', 'Find who took the farmer’s ledger', 'Every worker claims to have left before dusk', 'A fresh ladder mark reveals the ledger was hidden in a tree', '''Bea walked between the apple trees while Hollis repeated the workers' statements. Everyone had left before dusk; nobody admitted touching the ledger. At the oldest tree, Bea noticed a rectangle of fresh soil beneath a ladder. The ladder's top rung bore a strip of blue cloth. Hollis recognized the cloth from the farmer's own coat. They climbed and found the ledger wrapped in oilskin inside a hollow branch. The farmer admitted hiding it when a tax collector arrived. No thief had entered the orchard, but his lie had sent everyone searching.'''),
('The Lantern Ferry', 'horror', ['Sera','Benn'], 'A fog-covered river crossing', 'Bring a stranded child across the river before the ferry stops', 'A voice in the fog imitates Benn and lures Sera toward the water', 'Sera follows the real ferry bell and brings the child aboard', '''The ferry bell sounded twice beyond the fog. Sera held the stranded child's hand and called for Benn, who was waiting aboard. "This way," Benn's voice answered from the reeds. Sera took one step toward it, then stopped. Benn always rang the bell three times when the bank was hidden. The voice called again, closer now, but no bell followed. Sera turned toward the water and counted three clear strikes. A lantern appeared above the ferry's bow. She guided the child onto the deck without looking back at the reeds.'''),
('Paper Constellations', 'drama', ['Mei','Arlo'], 'A classroom after graduation', 'Convince Arlo to display his unfinished astronomy project', 'Arlo is ashamed that the final model is incomplete', 'Mei helps him label the missing pieces as future questions', '''The classroom had emptied except for a cardboard sky hanging from the ceiling. Arlo reached up to take it down. Half the paper stars were missing, and the painted planets had no labels. Mei caught the model before it fell. "The gaps are the interesting part," she said. She found blank cards and wrote questions beside each unfinished section: How do stars form? Why do planets tilt? Arlo added a question of his own. When the teacher returned, the model still had holes, but the cards made visitors stop and talk. Arlo left it hanging.'''),
('The Last Match', 'historical', ['Rafi','Nell'], 'A mountain signal post during a blizzard', 'Light the warning beacon before a supply convoy reaches an avalanche zone', 'Wet tinder and strong wind extinguish each attempt', 'They shelter a single match inside a metal cup and light the beacon', '''Snow had packed itself against the signal-post door. Rafi could hear the convoy bells somewhere below the ridge. Nell emptied the matchbox into her palm. One match remained. Their tinder was soaked, and the wind had killed every earlier flame. Rafi tore dry lining from his coat while Nell held a metal drinking cup around the match head. The flame caught the lining, then a twist of rope dipped in lamp oil. Together they carried it to the beacon basket. Orange light rose into the storm. Far below, the convoy bells stopped moving.'''),
('The Quiet Machine', 'sci_fi', ['Tess','Orin'], 'A lunar listening laboratory', 'Locate the source of an unexplained rhythmic signal', 'The laboratory instruments record nothing when the signal sounds', 'They discover vibrations traveling through the floor from a damaged pump', '''Tess heard three soft knocks through the soles of her boots. Orin checked the laboratory microphones; their displays were flat. Another three knocks came, exactly nine seconds later. Instead of recording the air, Tess placed her palm against the floor. The metal plates trembled beneath her fingers. They followed the vibration through a service corridor to a cooling pump whose loose mount struck the wall at each rotation. Orin tightened the bolts. The knocks ceased. Their mysterious signal had traveled through the moonbase itself, not through the silent lunar air.'''),
('A Crown of Reeds', 'fantasy', ['Ilse','Rowan'], 'A marsh kingdom at dawn', 'Return a ceremonial crown before the coronation', 'The stepping stones disappear under rising marsh water', 'Rowan builds a floating raft from reeds and they deliver the crown', '''The crown lay in Ilse's satchel, wrapped against the marsh mist. Across the water, the coronation bells began to ring. Rowan stepped toward the stone path, but the first marker vanished beneath the rising tide. Ilse could not swim with the heavy crown. Rowan gathered thick reeds, bound them with spare cord, and pushed the bundle into the shallows. It floated. They knelt together on the narrow raft and guided it with long branches. When they reached the palace landing, Ilse lifted the dry crown from her satchel. The bells had not yet finished their final peal.'''),
('The Painted Wall', 'contemporary', ['Dara','Luis'], 'A neighborhood community center', 'Restore a damaged mural before its opening ceremony', 'Rain has washed away the central portrait', 'They invite residents to add their own memories to the mural', '''Dara stood before the mural, watching rainwater drip from the blank oval where the founder's portrait had been. Luis checked the clock. The community center would open in an hour, and the paint could not dry in time. A neighbor arrived carrying a photograph of the founder planting the first tree. Then another brought a note about her cooking lessons. Dara pinned the photograph to the wall and asked everyone to write a memory on a colored card. By the time the doors opened, the missing portrait was surrounded by a hundred small stories.'''),
('The Sandglass Trial', 'thriller', ['Noor','Pavel'], 'An underground archive with a timed security door', 'Retrieve evidence before the archive seals permanently', 'The security timer advances whenever the lights flicker', 'They photograph the evidence and escape through the maintenance shaft', '''Noor found the sealed folder on the lowest shelf. Above the door, the timer showed forty seconds. Pavel pulled the folder free, but the lights flickered and the display jumped to twenty. There was no time to carry the heavy box of documents upstairs. Noor opened the folder and photographed each stamped page while Pavel loosened the grate over a maintenance shaft. The lights blinked again: eight seconds. They slid into the shaft with the photographs saved on Noor's camera. The steel door struck the floor behind them, trapping the paper but not the evidence.'''),
('The Second Recipe', 'slice_of_life', ['Amma','Leena'], 'A small kitchen before a family celebration', 'Recreate a lost family recipe', 'The original recipe card is stained beyond reading', 'They use remembered flavors to make a new version together', '''Leena held the recipe card toward the window, but the ink had dissolved into brown streaks. Amma smelled the toasted cumin and shook her head. "Your grandfather used more ginger," she said. They started again, tasting each spoonful before adding the next ingredient. Leena wrote down the changes on a clean card while Amma watched the pot. By evening the dish tasted familiar, though neither claimed it was identical to the old one. Leena titled the card Second Recipe and left space beneath it for future notes.'''),
]

# Genre normalization is restricted to genres supported by the project.
ALLOWED = {'action','adventure','fantasy','romance','mystery','thriller','horror','sci_fi','comedy','drama','slice_of_life','supernatural','historical','tragedy'}
GENRE_MAP = {'science fiction':'sci_fi','literary fiction':'drama','cozy fantasy':'fantasy','contemporary':'drama','historical fantasy':'fantasy','magical realism':'supernatural'}

def normalize_record(r):
    r = dict(r)
    genres = r.get('genres', [])
    genres = [genres] if isinstance(genres, str) else genres
    r['genres'] = [GENRE_MAP.get(str(g).lower(), str(g).lower()) for g in genres]
    if not r['genres'] or any(g not in ALLOWED for g in r['genres']):
        raise ValueError(f'Unsupported genre in {r.get("title")}: {r["genres"]}')
    return r

def validate(r):
    for key in ('title','genres','characters','setting','objective','obstacle','outcome','premise','story'):
        if not r.get(key): raise ValueError(f'Missing {key}: {r.get("title")}')
    import re
    for name in r['characters']:
        if not re.search(r'(?<!\w)'+re.escape(name)+r'(?!\w)', r['story'], re.I):
            raise ValueError(f'Character {name!r} absent in {r["title"]}')
    if len(r['story'].split()) < 65:
        raise ValueError(f'Story too short: {r["title"]}')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pilot-dir',default='dataset/v12/pilot')
    p.add_argument('--output-dir',default='dataset/v12/pilot_expanded')
    a=p.parse_args()
    pilot=Path(a.pilot_dir); out=Path(a.output_dir)
    if not all((pilot/f'{s}.jsonl').is_file() for s in ('train','val','test')):
        raise SystemExit('Missing original pilot files. Run: python -m training.v12.build_pilot')
    originals={}
    for split in ('train','val','test'):
        originals[split]=[normalize_record(json.loads(line)) for line in (pilot/f'{split}.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if [len(originals[s]) for s in ('train','val','test')] != [8,2,2]:
        raise SystemExit('Expected original pilot split sizes 8/2/2; refusing to overwrite or silently resplit.')
    fresh=[]
    for title,genre,characters,setting,objective,obstacle,outcome,story in NEW_SCENES:
        r={'title':title,'genres':[genre],'characters':characters,'setting':setting,'objective':objective,'obstacle':obstacle,'outcome':outcome,'premise':f'{objective}. {obstacle}. {outcome}.','story':story,'review_status':'AUTHOR_WRITTEN_PILOT','source_family':title}
        fresh.append(normalize_record(r))
    # Fixed, family-disjoint partition. No name/setting substitutions.
    splits={'train': originals['train']+fresh[:8], 'val':originals['val']+fresh[8:10], 'test':originals['test']+fresh[10:]}
    all_rows=[r for rows in splits.values() for r in rows]
    if len({r['title'] for r in all_rows}) != len(all_rows): raise ValueError('Duplicate titles')
    if len({r['story'] for r in all_rows}) != len(all_rows): raise ValueError('Duplicate stories')
    for r in all_rows: validate(r)
    out.mkdir(parents=True,exist_ok=True)
    for split,rows in splits.items():
        with (out/f'{split}.jsonl').open('w',encoding='utf-8') as f:
            for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    report={'total':len(all_rows),'splits':{k:len(v) for k,v in splits.items()},'status':'AUTHOR_WRITTEN_PILOT_NOT_PRODUCTION_VERIFIED','warning':'24 examples are inadequate for general-purpose instruction fine-tuning. Do not use for paid GPU training.','no_source_family_overlap':True,'v8_checkpoint_modified':False}
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
