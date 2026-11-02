# 転送検査と、まとめて完了を待つVulkan経路

このPixelでは64バイトの反復転送でも、非同期読み出しが前の値を返し、Vulkanのホスト領域へゼロを返すケースがあった。小さな再現例と数値ゲートを追加し、実験用ランタイムではキューの完了を明示的に待つ経路を選べるようにした。

## 使う

llama.cpp b11960/v0.8.0の同じソースから、CPU・Vulkan・llama・common・ツールをそろえたランタイムが必要。[ビルド手順](large-output.md#再開用のビルドと検査)の `--output-row-split` は、行分割と完了待ちのパッチ、4種類の検査ツールを組み込む。

    ./pixel-lab run inference --runtime /absolute/matched-runtime --gpu-layers 99 --split-output

`--split-output` はQwen2の大きな出力行列を分割し、同期読み出しとキュー完了待ちを自動選択する。既定の `--completion-wait queue` は、グラフをまとめて実行したまま、ホストが結果を読む前や領域を再利用する前にcompute queueの完了を待つ。

比較用の `--completion-wait serial` は、上流にある演算ごとの完了待ちを選ぶ。こちらは正しい値を確認する対照だが、VM経由の投入・待機回数が多く遅い。

    ./pixel-lab run inference --runtime /absolute/matched-runtime --gpu-layers 99 --split-output --completion-wait serial

CPU主体の対照は、出力層だけGPUへ置く要求を出し、行分割をせずに同期経路を使う。この端末・モデルでは大きな出力行列がCPUへfallbackする。

    ./pixel-lab run inference --runtime /absolute/matched-runtime --gpu-layers 1 --sync-transfers

ここでの条件はcontext 512、batch/ubatch 16、CPU 4スレッド、Flash Attention off、repack 1。異なるFlash Attention/repack条件は、この固定入力のモデル検査と一致しないため拒否する。

## 実機の数値と速度

2026-11-03、同じバイナリ群とQwen2.5-0.5Bファイルを使用。各行はpp64/tg32各3回の短い測定。ウォームアップなし、数値検査と配置確認を完了した後の測定で、長時間の持続性能を示すものではない。

| 条件 | tg32 平均 tok/s | pp64 平均 tok/s | プロセスのピークRSS KiB |
| --- | ---: | ---: | ---: |
| 全GPU要求＋出力行列分割、演算ごとに待つ対照 | 2.174 | 17.006 | 223840 |
| 全GPU要求＋出力行列分割、まとめて待つ新経路 | 8.144 | 39.694 | 223284 |
| CPU主体の対照、GPU要求1・出力行列はCPU | 35.537 | 68.754 | 607312 |

まとめて待つ経路は、演算ごとに待つ対照の約3.75倍のdecode速度だった。このモデルではCPU主体の経路がさらに速く、GPUへの行分割自体を高速化とは扱わない。RSSとVulkan割り当てには重なりがあり、足し合わせて総メモリや最大モデル容量の改善を主張しない。

固定33トークン入力の151936個のlogitは、全GPU要求の新経路と演算ごとに待つ対照でバイト単位に一致した。CPU参照に対するNMSEは約0.002246、最大絶対誤差は約0.836、top1は一致。事前条件のNMSE≤0.005とtop1一致を通した限定的な数値検査であり、回答品質や全プロンプトでの一致を保証しない。

最後の配置は全GPU要求でCPU 2ノード・Vulkan 611ノード。入力の埋め込み処理はCPUに残る。約137.94 MiBのQ8_0出力行列は3片のMUL_MATとCONCATになり、GPUに配置された。出力行列のみをGPUへ置く要求1でも数値ゲートを通した実行があるが、そちらの初期同期条件には後述の失敗もあり、安定した高速化として採用していない。

[各回の値・配置・失敗・全出力とハッシュ](../examples/inference/readback-study/)

## 何を変更したか

`GGML_VK_DISABLE_ASYNC=1` はb11960の既存機能で、グラフ完了を同期し、1次元の非同期get APIを同期読み出しへfallbackさせる。ただし、これだけでは後の反復検査で行列積が56件中1件不一致になった。

追加パッチは `GGML_VK_WAIT_IDLE=1` のとき、backendの同期点でcompute queueへ `vkQueueWaitIdle` を使う。既存のqueue/device-submitロックと同じ順序で排他する。演算ごとにCPUへ戻る方式の代わりに、グラフのまとめ実行を保って完了を確認する。[APIの完了待ち・ホスト排他の仕様](https://docs.vulkan.org/refpages/latest/refpages/source/vkQueueWaitIdle.html)

元の完了待ち経路と、この端末のdriver/VMの組み合わせがどう不一致を起こすかは断定していない。対策前後の値と待機方法の差を保存し、確認済みの実行条件として扱う。他端末・並列Vulkan contextでの性能や互換性は未評価。

新経路では転送27ケースと既存56ケースをそれぞれ3回照合し、その後もdoctorからモデル照合・配置・速度測定までを完走した。runtimeの対応フラグと全ライブラリ・検査ツールのハッシュがない場合、queue経路は使わない。

## 検査の順序と保存内容

1. doctorの259整数を照合。
2. 64バイト・4 KiB・1 MiB、オフセット64、異なる入力3回を3つの読み出し先／API経路で照合。27ケースの全バイトを比較し、欠落・重複も拒否。
3. F32・量子化行列積と正規化の既存56ケース。
4. 大きな出力行列のbasis入力検査。これは一般の密な総和の証明ではない。
5. 同じモデル・ランタイムでCPU参照とGPU候補の全logitを保存・比較。
6. 実際のCPU/GPU配置を記録し、条件が通ってから速度を測定。

`check-readback` はモデルやshader graphを作らない。Vulkanのホストバッファ型を要求するが、割り当てが通常メモリへfallbackする可能性は残る。2次元転送やあらゆるサイズの保証には使わない。JSON・生ログ・全出力・ハッシュを残し、失敗した検査の次へ進まない。対照のモデル検査で行分割を勝手に有効化しない。

非同期有効、同期getのみ、FP16無効、fusion停止、batch=1の先行診断も保存した。同期getのみの2組目はカーネル検査で止まり、速度行が存在しない。失敗した測定をやり直した成功行に置き換えていない。先行する9 tok/s前後の値は今回の完了待ち・全logitゲートを通していないため、改善率の対照には使わない。

## このPixelで再開

確認済みファイルを保存する場所は `~/.local/share/SpiralPixcel/runtimes/llama-b11960-queue-wait`。古いcore-fp16ランタイムも残す。Macからは次のように呼ぶ。

    ./pixel-lab run inference --runtime /home/droid/.local/share/SpiralPixcel/runtimes/llama-b11960-queue-wait --gpu-layers 99 --split-output

任意の表示設定・Google同期停止・期限付き実験と復旧は[実験用セットアップ](lab-session.md)。自動ロックの解除を必須条件にせず、今回の実機検証もロック中の短いセッションで継続した。最大モデル容量、長いcontext、LoRA、実用的な回答品質、持続負荷は今後の検証対象として残す。
