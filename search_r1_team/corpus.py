"""Read JSONL corpus rows without loading the entire text collection into RAM."""
import json
import mmap
from pathlib import Path
import struct

class IndexedCorpus:
    def __init__(self,path):
        self.path=Path(path); offsets=self.path.with_suffix(self.path.suffix+'.offsets')
        meta=offsets.with_suffix(offsets.suffix+'.json')
        stat=self.path.stat(); signature={'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}
        if not offsets.exists() or not meta.exists() or json.loads(meta.read_text())!=signature:
            temp=offsets.with_suffix('.building')
            with self.path.open('rb') as source,temp.open('wb') as target:
                while True:
                    position=source.tell(); line=source.readline()
                    if not line: break
                    if line.strip(): target.write(struct.pack('<Q',position))
            temp.replace(offsets); meta.write_text(json.dumps(signature))
        self.handle=self.path.open('rb'); self.offsets_handle=offsets.open('rb')
        if offsets.stat().st_size==0: self.close(); raise ValueError('Corpus is empty')
        self.index=mmap.mmap(self.offsets_handle.fileno(),0,access=mmap.ACCESS_READ)
        self.count=len(self.index)//8
    def __len__(self): return self.count
    def __getitem__(self,index):
        if not 0<=index<self.count: raise IndexError(index)
        offset=struct.unpack_from('<Q',self.index,index*8)[0]
        self.handle.seek(offset)
        doc=json.loads(self.handle.readline())
        if not isinstance(doc.get('contents'),str): raise ValueError('Corpus row needs contents')
        return {'id':str(doc.get('id',index)),'contents':doc['contents']}
    def close(self):
        for name in ('index','offsets_handle','handle'):
            value=getattr(self,name,None)
            if value is not None: value.close()
