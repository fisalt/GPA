import itertools
import torch
from gpa.finetune import PackedTextDataset,packed_collator


class Tokenizer:
    eos_token_id=99
    def __call__(self,text,add_special_tokens=False):
        return {"input_ids":[int(v) for v in text.split()]}


def test_packing_documents_eos_targets_and_drop_partial():
    rows=[{"text":"1 2 3"},{"text":"4 5 6 7 8"},{"text":"9"}]
    data=PackedTextDataset(Tokenizer(),5,rows=rows,repeat=False,shuffle_buffer=0)
    batches=list(data)
    assert [b["input_ids"].tolist() for b in batches]==[[1,2,3,99,4],[5,6,7,8,99]]
    batch=packed_collator(batches)
    assert torch.equal(batch['labels'],batch['input_ids'])
    assert batch['attention_mask'].bool().all()
    assert batch['labels'].data_ptr()!=batch['input_ids'].data_ptr()


def test_stream_replay_and_accelerate_single_owner_sharding():
    rows=[{"text":str(i)} for i in range(60)]
    source=lambda:PackedTextDataset(Tokenizer(),4,rows=rows,repeat=False,shuffle_buffer=0)
    a=list(source());b=list(source())
    assert all(torch.equal(x['input_ids'],y['input_ids']) for x,y in zip(a,b))
    # Each consecutive global batch is split once, never pre-sharded again.
    import pytest
    try:from accelerate.data_loader import IterableDatasetShard
    except ImportError:pytest.skip("Install hf extra to test distributed data sharding")
    left=list(IterableDatasetShard(source(),batch_size=1,num_processes=2,process_index=0,drop_last=True))
    right=list(IterableDatasetShard(source(),batch_size=1,num_processes=2,process_index=1,drop_last=True))
    assert len(left)==len(right)==len(a)//2
    assert all(torch.equal(x['input_ids'],a[2*i]['input_ids']) for i,x in enumerate(left))
    assert all(torch.equal(x['input_ids'],a[2*i+1]['input_ids']) for i,x in enumerate(right))


def test_local_parquet_stream_and_packing(tmp_path):
    import pytest
    pytest.importorskip('datasets')
    pa=pytest.importorskip('pyarrow')
    import pyarrow.parquet as pq
    path=tmp_path/'texts.parquet'
    pq.write_table(pa.Table.from_pylist([{'text':'1 2 3'},{'text':'4 5 6 7 8'}]),path)
    stream=PackedTextDataset(Tokenizer(),5,data_files=[str(path)],repeat=False,shuffle_buffer=0)
    assert [v['input_ids'].tolist() for v in stream]==[[1,2,3,99,4],[5,6,7,8,99]]
