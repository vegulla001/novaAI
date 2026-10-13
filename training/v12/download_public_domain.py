"""Download substantial *prose* corpus from Project Gutenberg (NOT instruction pairs).
Run from repository root: python -m training.v12.download_public_domain
Verify public-domain status in your jurisdiction before use.
"""
import argparse, json, re, time, urllib.request, urllib.error
from pathlib import Path

BOOKS = [
(11,'Alice’s Adventures in Wonderland','fantasy'),(84,'Frankenstein','horror'),
(1342,'Pride and Prejudice','romance'),(1661,'The Adventures of Sherlock Holmes','mystery'),
(2701,'Moby Dick','adventure'),(98,'A Tale of Two Cities','historical fiction'),
(345,'Dracula','horror'),(174,'The Picture of Dorian Gray','horror'),
(76,'Adventures of Huckleberry Finn','adventure'),(1400,'Great Expectations','historical fiction'),
(55,'The Wonderful Wizard of Oz','fantasy'),(120,'Treasure Island','adventure'),
(35,'The Time Machine','science fiction'),(36,'The War of the Worlds','science fiction'),
(1260,'Jane Eyre','romance'),(514,'Little Women','historical fiction'),
(2591,'Grimms’ Fairy Tales','fantasy'),(43,'The Strange Case of Dr Jekyll and Mr Hyde','horror'),
(730,'Oliver Twist','historical fiction'),(158,'Emma','romance'),
(74,'The Adventures of Tom Sawyer','adventure'),(768,'Wuthering Heights','romance'),
(1497,'The Republic','other'),(16328,'Beowulf','fantasy'),
(209,'The Turn of the Screw','horror'),(46,'A Christmas Carol','fantasy'),
(105,'Persuasion','romance'),(996,'Don Quixote','adventure'),
(1259,'Twenty Thousand Leagues Under the Seas','science fiction'),
(16,'Peter Pan','fantasy'),(521,'The Life and Adventures of Robinson Crusoe','adventure'),
(203,'Uncle Tom’s Cabin','historical fiction'),
]
START = re.compile(r'\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\*\*', re.I)
END = re.compile(r'\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\*\*', re.I)

def strip_gutenberg(text):
    s=START.search(text);e=END.search(text)
    if not s or not e or e.start() <= s.end():
        raise ValueError('Gutenberg START/END markers missing; refusing to include license/header text')
    body=text[s.end():e.start()].strip()
    body=re.sub(r'\r\n?', '\n',body)
    body=re.sub(r'\n{4,}', '\n\n\n',body)
    return body

def fetch(book_id,timeout):
    urls=[f'https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8',
          f'https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt',
          f'https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt',
          f'https://www.gutenberg.org/files/{book_id}/{book_id}.txt']
    for url in urls:
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'StoryAI research corpus downloader (personal educational use)'})
            with urllib.request.urlopen(req,timeout=timeout) as r:
                raw=r.read(20_000_000)
            return strip_gutenberg(raw.decode('utf-8-sig',errors='replace')),url
        except (urllib.error.URLError,TimeoutError,ValueError,UnicodeError) as ex:
            last=ex
    raise RuntimeError(f'No usable UTF-8 Gutenberg text for {book_id}: {last}')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='dataset/v12/public_domain')
    p.add_argument('--max-books',type=int,default=len(BOOKS))
    p.add_argument('--min-words',type=int,default=150)
    p.add_argument('--max-words',type=int,default=350)
    p.add_argument('--timeout',type=int,default=25)
    args=p.parse_args()
    if not 1<=args.max_books<=len(BOOKS) or not 30<=args.min_words<=args.max_words:
        p.error('Invalid arguments')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    rows=0;books_ok=0;failures=[];manifest=[]
    with (out/'prose_segments.jsonl').open('w',encoding='utf-8') as fp:
        for bid,title,genre in BOOKS[:args.max_books]:
            try:
                text,url=fetch(bid,args.timeout)
                (out/f'book_{bid}.txt').write_text(text,encoding='utf-8')
                # Paragraph-aligned segments; no invented instruction labels.
                paras=[p.strip().replace('\n',' ') for p in re.split(r'\n\s*\n',text) if p.strip()]
                buffer=[];word_count=0;segments=0
                for para in paras:
                    words=para.split()
                    if len(words)>args.max_words:
                        chunks=[words[i:i+args.max_words] for i in range(0,len(words),args.max_words)]
                    else: chunks=[words]
                    for chunk in chunks:
                        if buffer and word_count+len(chunk)>args.max_words:
                            if word_count>=args.min_words:
                                fp.write(json.dumps({'text':' '.join(buffer),'book_id':bid,'title':title,'genre':genre,'source_url':url,'type':'public_domain_prose_not_instruction'},ensure_ascii=False)+'\n')
                                rows+=1;segments+=1
                            buffer=[];word_count=0
                        buffer.extend(chunk);word_count+=len(chunk)
                if word_count>=args.min_words:
                    fp.write(json.dumps({'text':' '.join(buffer),'book_id':bid,'title':title,'genre':genre,'source_url':url,'type':'public_domain_prose_not_instruction'},ensure_ascii=False)+'\n')
                    rows+=1;segments+=1
                books_ok+=1;manifest.append({'id':bid,'title':title,'genre':genre,'source_url':url,'segments':segments})
                print(f'OK {bid} {title}: {segments} segments',flush=True)
            except Exception as exc:
                failures.append({'id':bid,'title':title,'error':str(exc)})
                print(f'SKIP {bid} {title}: {exc}',flush=True)
            time.sleep(0.5)
    (out/'manifest.json').write_text(json.dumps({'books':manifest,'failures':failures,'segments':rows,'warning':'Prose pretraining only; NOT V12 instruction/story pairs. Check rights locally.'},indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'Finished: {books_ok} books, {rows} prose segments, {len(failures)} failures. See {out}/manifest.json')
if __name__=='__main__':main()
