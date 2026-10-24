# The inference protocol (RMIP v2, plan step 5d)

How an actor (`nn:remote:<endpoint>`, `src/actor/remote_evaluator.h`) asks an inference server for
the evaluations of featurized decisions. Framework-neutral and versioned: a server is anything that
speaks it. `rmserve` (`tools/rmserve/`, C++, libtorch, CPU) is the reference server;
`tools/nn/serve_check.py` is a minimal one in Python (`socket`, `struct`, `json`,
`multiprocessing.shared_memory`, numpy and torch), and the training plan's GPU server
(`docs/rnad_training_plan.md` L5) implements the same protocol. The C++ wire code (framing, the
request and response layouts, the shared region) is `src/inference/protocol.h`, standard library and
POSIX only.

Version 2 (`docs/open_vocab_plan.md` S3) adds the tables' layout hash and vocab manifest to the
handshake (`HELLO`'s `layout_hash`, `tables_digest` and `vocab`, §3): a server's exports are built for one vocab and
tables, and a client on other tables is refused naming what differs (an export of the vocab before a
card was appended: rebuild it from its checkpoint). Requests and responses are as in version 1 but for
the protocol number.

## 1. Transport

- **Control:** a Unix domain stream socket (`AF_UNIX`, `SOCK_STREAM`) at the endpoint path. One
  connection per client; a process with two `nn:remote:` seats opens two.
- **Payloads:** a POSIX shared-memory region per connection (`shm_open` + `mmap`), created by the
  client and named in its handshake. It holds a **request area** (offset 0, `request_bytes`) and a
  **response area** (offset `request_bytes`, `response_bytes`). The socket carries only small
  messages; tensors never travel over it.
- **One request in flight per connection.** The client writes a request into the request area,
  sends `EVAL` and waits; the server reads the request, evaluates it (possibly batched with other
  connections' requests, §5), writes the response into the response area and sends `DONE`. Only
  then may the client touch the region again. A client that wants several requests in flight opens
  several connections.
- Every integer and float is little-endian; floats are IEEE binary32.

## 2. Control messages

Each message is a 12-byte header and a body:

| Offset | Type | Field |
|---|---|---|
| 0 | 4 bytes | magic `RMIP` |
| 4 | u16 | message type |
| 6 | u16 | reserved, 0 |
| 8 | u32 | body length in bytes (at most 1 MiB) |

| Type | Name | Direction | Body |
|---|---|---|---|
| 1 | `HELLO` | client → server | JSON object (§3) |
| 2 | `WELCOME` | server → client | JSON object (§3) |
| 3 | `EVAL` | client → server | u64 `seq`, u64 request length: a request is in the request area (§4) |
| 4 | `DONE` | server → client | u64 `seq`, u64 response length: its response is in the response area (§4) |
| 5 | `ERROR` | server → client | JSON object `{"code": …, "message": …}` (§6); the server then closes the connection |
| 6 | `BYE` | client → server | empty: the client is done (closing the socket means the same) |

In Python: `struct.pack("<4sHHI", b"RMIP", type, 0, len(body)) + body`.

## 3. Handshake

The client creates the region (`shm_open(name, O_CREAT | O_EXCL | O_RDWR, 0600)`, `ftruncate` to
`request_bytes + response_bytes`), connects and sends `HELLO`, one JSON object:

| Key | Value |
|---|---|
| `protocol` | 2 |
| `client` | a free-form name for the server's log (`robomage_nn pid 4242 seat A`) |
| `spec_hash`, `vocab_hash` | the featurizer's `FEATURE_SPEC` hash and the vocab hash, 16 lowercase hex digits each (`docs/featurizer.md` §6) |
| `text_model` | the text table's encoder (`<repository>@<commit>`) |
| `layout_hash` | the layout registry's `layout_hash`, 16 lowercase hex digits (`docs/featurizer.md` §6.1) |
| `tables_digest` | the tables' digest (every card's card, ability and part rows), 16 lowercase hex digits (`docs/featurizer.md` §6.2) |
| `vocab` | the vocab manifest: an array of `[id, name]`, one per card-table row (`docs/featurizer.md` §6.2) |
| `dims` | `{"F": …, "C": …, "K": …, "Q": …}`, the spec's widths (§4) |
| `shm` | the region's POSIX name, starting with `/` |
| `request_bytes`, `response_bytes` | the two areas' sizes, each a multiple of 8 |
| `policies` | the policy ids to evaluate on every request, non-empty and distinct; results come back in this order. Policy 0 is the one an actor samples from ("current"); the training plan's regularisation policies are other ids |

The server checks that it serves every requested policy and that each one's model is an export of
the client's tables exactly (`docs/featurizer.md` §6.2, `check_compatible` exact: the same
`layout_hash` and `text_model`, the same vocab manifest — no id the model's lacks, none removed or
renamed — and the same `spec_hash`, `vocab_hash` and `tables_digest`; the model's metadata, `docs/featurizer.md` §6),
opens and maps the region (its size at least the two areas), and answers `WELCOME`:

| Key | Value |
|---|---|
| `protocol` | 2 |
| `server` | a free-form name (`rmserve`) |
| `max_batch` | the most decisions the server evaluates in one batch (informative) |
| `policies` | per requested policy, in the client's order, `{"id": …, "version": …}` |

or `ERROR` and closes. On either answer the client unlinks the name (`shm_unlink`): both sides keep
their mappings, and a client that dies leaks nothing. A server needs nothing but the client's
`dims` to read requests (it never needs the feature spec document); the spec hash ties the two.

## 4. Requests and responses

**Request** (the request area). A header, the policy ids, the per-decision sizes, then the
decisions' unpadded tensors concatenated decision after decision, exactly the arrays of an RMTB
`tensors` file (`docs/featurizer.md` §3-4: same names, order, dtypes and per-decision local
indices), with each decision's sizes in place of the `_offsets` arrays (an array's offsets are the
cumulative sums of its decisions' sizes). Every array starts at a multiple of 8 bytes from the
area's start (zero padding between).

| Offset | Type | Field |
|---|---|---|
| 0 | 4 bytes | magic `RMRQ` |
| 4 | u32 | protocol, 2 |
| 8 | u64 | `spec_hash` (on every request; the server refuses a mismatch) |
| 16 | u64 | `seq`: the client's request counter, echoed in `DONE` and the response |
| 24 | u32 | `D`, decisions (≥ 1) |
| 28 | u32 | `P`, policies (≥ 1; the handshake's list) |
| 32 | u32 × 4 | `F`, `C`, `K`, `Q` (the handshake's `dims`) |
| 48 | u64 | the request's length in bytes, this header included |
| 56 | u64 | reserved, 0 |
| 64 | u32 × P | policy ids, the handshake's list in its order |

then, each at the next multiple of 8:

| Array | Dtype | Values |
|---|---|---|
| `n` | i64 | `[D]` tokens per decision (≥ 1) |
| `r` | i64 | `[D]` relations per decision (≥ 0) |
| `a` | i64 | `[D]` menu rows per decision (≥ 1) |
| `tok_type` | i64 | `[ΣN]` |
| `tok_num` | f32 | `[ΣN, F]` |
| `tok_cat` | i64 | `[ΣN, C]` |
| `tok_card` | i64 | `[ΣN, K]` (-1 an unused slot) |
| `tok_ability` | i64 | `[ΣN, Q]` (-1 an unused slot) |
| `tok_mask` | u8 | `[ΣN]` |
| `rel` | i64 | `[ΣR, 3]` (src, dst, type; src and dst index the decision's own tokens) |
| `act_tok` | i64 | `[ΣA]` (menu row → the decision's own token) |
| `act_mask` | u8 | `[ΣA]` |

The request ends with the last array (its length is that end). A server pads and stacks the
decisions into the network contract's `[B, …]` inputs as it likes (`docs/featurizer.md` §6:
`rel_mask` marks each decision's `r` real relations); the reference servers pad as
`src/featurize/batch.h` does.

**Response** (the response area):

| Offset | Type | Field |
|---|---|---|
| 0 | 4 bytes | magic `RMRS` |
| 4 | u32 | protocol, 2 |
| 8 | u64 | `spec_hash`, the request's |
| 16 | u64 | `seq`, the request's |
| 24 | u32 | `D` |
| 28 | u32 | `P` |
| 32 | u32 | `batch`: decisions in the server batch this request was evaluated in (diagnostic) |
| 36 | u32 | `compute_us`: that batch's evaluation time in microseconds, saturating (diagnostic) |
| 40 | u64 | the response's length in bytes |
| 48 | u64 | reserved, 0 |
| 56 | (u32, u32, u64) × P | per policy, in the request's order: id, reserved 0, **version** |

then per policy, in the same order, each at the next multiple of 8: `value` f32 `[D]`, `logits` f32
`[ΣA]` and `log_pi` f32 `[ΣA]`, the decisions' menu rows concatenated as in the request. `logits`
are the policy's over each decision's real menu rows only; `log_pi` is their log-softmax over those
rows; `value` is the policy's value for the deciding seat. The version is the policy's weights'
version at the time of this response, so a server may swap weights between requests and every
response says which it used.

## 5. Batching

A server may hold a request to evaluate it together with other connections' requests, up to a
batch size or a deadline (a micro-timeout); it never splits a request, and it answers each
connection's requests in order. `rmserve` takes queued requests in arrival order while their
decisions fit `--max-batch` (a larger request is a batch of its own) and evaluates a batch when the
queued decisions reach `--max-batch`, when every connected client has a request queued (nobody else
can add to it), or `--deadline-us` after the oldest queued request arrived. Each policy runs once
per batch, over the decisions that asked for it.

## 6. Errors

Every refusal is an `ERROR` message, after which the server closes the connection. `code`:

| Code | When |
|---|---|
| `protocol` | a malformed message or JSON, an unknown type, an unsupported `protocol` |
| `layout_hash`, `text_model`, `vocab_hash`, `spec_hash`, `tables_digest` | a requested policy's model is not an export of the handshake's tables: the first that differs in that order (`vocab_hash` for the vocab manifest too: an id the client has and the export lacks, or one removed or renamed; the message says to rebuild the export); `spec_hash` also a request whose header differs from the handshake's |
| `policy` | a requested policy the server does not serve |
| `shm` | the region cannot be opened or mapped, or is smaller than its areas |
| `request` | a request that is not the layout above: bad magic, protocol, dims or policies; sizes out of bounds or past the request area; a length that does not match; an `act_tok` or `rel` index outside its decision's tokens |
| `eval` | the model failed on the batch (every request of the batch gets it) |
| `shutdown` | the server is stopping |

A client treats an `ERROR`, the connection closing, a response that is not its request's (`seq`,
`D`, `P`, policy ids, spec hash) and no answer within its timeout as fatal: the `nn:remote:` actor
exits naming the endpoint, never taking a default action.

## 7. Implementing a server

In Python (`tools/nn/serve_check.py` is a complete example, about 200 lines):

```python
hdr = struct.unpack_from("<4sIQQII4IQQ", shm.buf, 0)        # magic … reserved
D, P, F, C, K, Q = hdr[4], hdr[5], *hdr[6:10]
off = align8(64 + 4 * P)
n = np.frombuffer(shm.buf, np.int64, D, off); off = align8(off + 8 * D)   # then r, a, the tensors
```

then for each array `np.frombuffer(shm.buf, dtype, count, off)` with `count` from the sizes' sums,
split per decision with `np.cumsum`, pad, run the model, and write the response with
`struct.pack_into` and numpy views of the response area. Open the region with
`multiprocessing.shared_memory.SharedMemory(name.lstrip("/"), track=False)` (Python 3.13; without
`track=False` the resource tracker unlinks it at exit).
