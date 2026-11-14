# doctorからモデル推論へ

`pixel-lab run inference` は同じVMで整数GPUチェックを実行した後、llama.cppのpp64・tg32を測る。既存のQwen2.5-0.5B Q4_K_Mとcore16修正版を既定値にする。

    pixel-lab run inference --repetitions 3

`--runtime` と `--model` はDebian VM内の絶対パス。`--gpu-layers 8` などでGPUに置く層数を調整できる。条件はバッチ16、マイクロバッチ16、CPU4スレッド、既定のFlash Attention off、ウォームアップなし。各テスト3回。小さい入力での測定として扱う。

結果は既存の `runs/<ID>/` に残る。`doctor.json`、`bench.json`、`bench.log`、`result.json` に実行条件、各回の速度、モデルとライブラリのSHA-256、20ms間隔のプロセスRSSとVulkan割り当て量を保存する。RSSとVulkan割り当てには重なりがあり、合算しない。推論速度の測定完了とモデルの回答品質は別に評価する。

## PowerVRカーネルを比較する

パッチはllama.cpp b11960 / v0.8.0のソース向け。既存ランタイムを保存して別の出力先にビルドする。DebianにCMake、Ninja、C++、glslc、libvulkan-dev、Vulkan-Hppが必要。

    sh guest/build-inference-runtime.sh /absolute/fresh/llama.cpp-0.8.0 /absolute/stock-runtime /absolute/new-runtime

コンパイルのメモリ消費を抑えるためC/C++はO1、ビルドは1並列。SPIR-Vは既存シェーダーと実行時の特殊化定数を使う。計測とビルドは順番に行う。

- core16はコア機能に昇格済みの拡張名を要求しない起動修正。
- core-fp16はVulkan 1.1以上の16ビットストレージと実際の機能ビットを確認し、FP16計算を使えるようにする。
- powervr-mmvは共有メモリでの縮約とワークグループ32・64・128を選べる実験。
- powervr-rmsは既存のサブグループ正規化をPowerVRでも選べる実験。
- powervr-u8-mmvqは、Q5_0・Q8_0・Q4_K・Q6_Kの1列mat-vecを符号なしdot積で計算する。`--powervr-u8-mmvq` でビルドし、`--u8-mmvq` で選ぶ。同じランタイムで切り替えた交互測定で、decode tg32が平均8.06から18.74 tok/sになった（Q6_Kを加える前の版）。[測り方と結果](kernels.md)。Q5_0とQ8_0は行数で変種を選ぶ（[形ごとの変種](matvec-variants.md)）。
- powervr-decodeは、1トークンのattentionスコアの専用シェーダー、計算どうしの安いバリア、device-localのバッファ配置を足す。`--powervr-decode`（`--output-row-split` と `--powervr-u8-mmvq` が必要）でビルドし、`--powervr-decode` で選ぶ。同じランタイムの交互測定で、decode tg32が平均18.89から28.56 tok/sになった。[測り方と結果](decode.md)

    pixel-lab run inference --runtime /absolute/new-runtime
    pixel-lab run inference --runtime /absolute/new-runtime --workgroup 64
    pixel-lab run inference --runtime /absolute/new-runtime --rms-subgroups
    pixel-lab run inference --runtime /absolute/new-runtime --flash-attn on

workgroupやrmsを指定する場合はライブラリのビルド記録を検査し、CPUとVulkanのテンソル照合を通してからベンチマークに進む。F32・Q4_K・Q5_0・Q6_Kの行列積と、正規化および乗算グラフを検査する。量子化済みの同じ重みを両バックエンドに渡す。閾値と実際の誤差は `kernel-check.jsonl` に残る。

`--profile host-memory` は割り当て先の優先順、`--profile mmvq` は既存の量子化ベクトル積経路を選ぶ比較用。全プロファイルを性能と数値の両面で確認してから採用する。FP16の無効化やワークグループ変更は対象プロセスの環境に適用する。

## 先行する実機結果

2026-11-03、Pixel 11 Proの標準Debian VM、PowerVR、同じb11960を使用した。短い測定であり、長時間の持続性能や最大モデル容量は未確定。

| 条件 | tg32 平均 tok/s | ピークRSS KiB |
| --- | ---: | ---: |
| 既存0.5Bファイル、core-fp16、初回 | 9.14 | 374940 |
| 同じライブラリでFP16無効 | 8.20 | 395680 |
| FP16有効を再測定 | 8.12 | 413700 |
| 1.5B Q4_K_M、GPU8層、repackなし（再開用ランタイム） | 12.71 | 1032888 |

FP16有効化は56ケースのCPU/Vulkan照合を通した。一方、速度とRSSの優位は再測定で維持されなかった。ワークグループ32/64/128、正規化のサブグループ化、行の処理数を増やす候補も調べ、今回の範囲では安定したdecode向上を確認できなかった。既定のランタイムに高速化を主張する変更は入れていない。

1.5Bモデルのファイルサイズは1,117,320,736バイト。GPU8層とCPUを併用し、pp64/tg32各3回を完走した。「赤い惑星」を尋ねる単発生成ではMarsを返した。これは同一モデルの高速化比較とは分けて扱う。

[測定JSONと各回の値](../examples/inference/) · [56ケースの数値照合](../examples/inference/kernel-check.jsonl) · [モデルのリビジョンとSHA-256](../examples/models.json)

### このPixelで再開する

検証用ランタイムはVMの `~/.local/share/SpiralPixcel/runtimes/llama-b11960-core-fp16`、モデルは `~/.local/share/spiral-pixel/models/` にある。Macから次で同じ設定を測れる。

    pixel-lab run inference --runtime /home/droid/.local/share/SpiralPixcel/runtimes/llama-b11960-core-fp16 --model /home/droid/.local/share/spiral-pixel/models/qwen2.5-1.5b-instruct-q4_k_m.gguf --gpu-layers 8 --repack 0 --repetitions 3

標準ターミナルで単発生成する例。

    rt="$HOME/.local/share/SpiralPixcel/runtimes/llama-b11960-core-fp16"
    LD_LIBRARY_PATH="$rt" "$rt/llama-completion" -m "$HOME/.local/share/spiral-pixel/models/qwen2.5-1.5b-instruct-q4_k_m.gguf" -dev Vulkan0 -ngl 8 --no-repack -fit off -b 16 -ub 16 --no-warmup -fa off -c 512 -t 4 -n 64 --temp 0 -p 'Write one sentence about a spiral.'

### メモリと次の切り分け

Androidがビルドとベンチマークの同時実行中にMemoryLimiter:AnonSwapでターミナルを終了した。VM内のfree値だけではAndroid側の制限を把握できない。今回は1並列ビルドと推論を順番に行って再開した。

このVMのmaxStorageBufferRangeは128 MiB。b11960はこの範囲を超えるテンソルを、64ビットインデックス経路がない場合にGPU対象から外す。大きい出力行列の分割、CPUに残る処理の特定、長いKVキャッシュの測定を次の実験として残す。最大モデル容量の増加や、特定のCPUフォールバックがボトルネックであるという結論は、まだ実測していない。

上記は先行手順の記録で、後続の完了待ち・全logitゲートを通した数値ではない。新モードとの改善率の対照には使わない。

続く調査で、0.5Bファイルの最終行列積がCPUに残ることを特定した。[大きな出力行列の調査](large-output.md)と[完了待ち修正の実機結果](readback.md)に、失敗からの切り分けを残した。完了待ちの修正で数値ゲートを通し、逐次待ちの対照よりdecodeを改善した。行列分割自体の高速化や、最大モデル容量の増加は実証していない。
