import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import urllib.request
import re
import html
import json
from pathlib import Path
from collections import defaultdict
import plotly.graph_objects as go


# ------- helper func --------

def is_exist_config():
    """
    config.json から max_or_min の値を取得する。存在しない、または不正な値の場合は None を返す。
    """
    config_path = Path(__file__).resolve().parent / "config.json"
    if not config_path.exists():
        return None
    
    # maxなのかminなのか返すようにする
    with config_path.open("r",encoding="utf-8") as f:
        config = json.load(f)
    return config["max_or_min"] if config.get("max_or_min") in ["max","min"] else None


def load_config():
    """
    config.json を読み込み辞書で返す。存在しない場合は空辞書を返す。
    """
    config_path = Path(__file__).resolve().parent / "config.json"
    if not config_path.exists():
        return {}
    
    with config_path.open("r",encoding="utf-8") as f:
        return json.load(f)


def fetch_visualizer(URL:str):
    #URLからvisualizerのHTMLを取得する。
    if URL == "":
        return None
    if URL.startswith("https://"):
        try:
            with urllib.request.urlopen(URL) as res:
                raw = res.read().decode("utf-8")

                #中身だけ抜き取る
                m = re.search(r'data-code="([^"]+)"',raw,re.DOTALL)
                if not m:return None

                escaped_html = m.group(1)
                decoded_html = html.unescape(escaped_html)
                return decoded_html

        except:
            return None
    return None


# logs フォルダのパスを定義
logs_dir = Path(__file__).resolve().parent.parent / "logs"


def fetch_data_from_logs(logs_dir = logs_dir):
    #logsフォルダからデータを引っ張り出す関数
    # -> list[tuple[YYYYMMDDhhmmss,フォルダのpath]]

    #logsフォルダが存在するかどうかの判定
    if not logs_dir.exists():
        return None
    
    pattern = re.compile(r"^(.*)_(\d{8})_(\d{6})$")
    folders = []

    for p in logs_dir.iterdir():
        if not p.is_dir():
            continue

        m = pattern.match(p.name)
        if not m:continue

        comments,yyyymmdd,hhmmss = m.groups()
        timestamp = yyyymmdd + hhmmss
        folders.append((timestamp,p))
    
    folders.sort(key = lambda x: x[0],reverse = True) #新しい順にソート

    if len(folders) == 0: #そもそも中身がなかった場合
        return None
    return folders


def fetch_data_from_floders(folders):
    #foldersの中身から、seeds,各seed毎のスコア,時間,comment,総合スコア,AC数/全体数を返すリストを作成する
    # ->{path:[[seeds],[各seed毎のスコア]YYYY-MM-DD hh:mm:ss,comment,総合スコア,AC数/全体数}

    details = {}
    for YYYYMMDDhhmmss,path in folders:
        array_of_data = []
        #日付のフォーマットを整える(YYYY-MM-DD hh:mm:ss)
        date = YYYYMMDDhhmmss[:4]+"-"+YYYYMMDDhhmmss[4:6] + "-" + YYYYMMDDhhmmss[6:8] + " " + YYYYMMDDhhmmss[8:10] + ":" + YYYYMMDDhhmmss[10:12] + ":" + YYYYMMDDhhmmss[12:]

        #コメントを取得する
        #コメントがデフォルトの"run"の時は時刻がないと区別できないので、時刻もコメントに入れておく
        pattern = re.compile(r"^(.*)_(\d{8})_(\d{6})$")
        m = pattern.match(path.name)
        comment,_,_ = m.groups()
        
        if comment == "run":
            comment = comment + "-"+  YYYYMMDDhhmmss

        #results.jsonからseed,スコア,AC数を求める
        seeds = []
        scores = []
        sum_score = 0
        AC_count = 0
        
        results_path = path / "results.json"
        if not results_path.exists():
            return None
        
        with results_path.open("r",encoding = "utf-8") as f:
            data = json.load(f)
        
        for d in data:
            seed = d["seed"]
            # seed は数値で扱う。文字列の場合は int に変換しておく。
            try:
                seed = int(seed)
            except Exception:
                pass
            read_score = d["scorer_score"]
            if read_score is None:#TLE
                score = 0
            elif read_score == -1:#WA
                score = 0
            else:
                score = int(read_score)
                AC_count += 1
            
            scores.append(score)
            sum_score += score
            seeds.append(seed)

            # shared_vars を収集（results.json の各エントリごとに存在する可能性がある）
            # shared_list は seeds と同じ順序に揃える
            try:
                shared = d.get("shared_vars", {})
            except Exception:
                shared = {}
            if 'shared_list' not in locals():
                shared_list = []
            shared_list.append(shared)
        
        # shared_list が存在しない場合は空リストを格納しておく
        if 'shared_list' not in locals():
            shared_list = []

        details[path] = [seeds,scores,date,comment,sum_score,str(AC_count)+"/"+str(len(seeds)), shared_list]
    return details


def format_score_scientific(value: float) -> str:
    """
    スコアや合計値を有効数字3桁の指数表記文字列にする。
    """
    try:
        return f"{float(value):.3e}"
    except:
        return str(value)


def catmull_rom_chain(xs, ys, points_per_segment=200):
    """
    Catmull-Rom スプラインで ys を補間する。
    xs: 元の x 座標（等間隔のインデックスでも可）
    ys: 元の y 値
    points_per_segment: 各セグメントあたりに生成する点数
    戻り値: (xnew_array, ynew_array)
    """
    n = len(xs)
    if n < 2:
        return np.array(xs), np.array(ys)

    # xs は等間隔インデックスで扱う（positions を渡す想定）
    positions = list(range(n))

    # 拡張配列を作る（端点を複製）
    ys_ext = [ys[0]] + list(ys) + [ys[-1]]

    xnew = []
    ynew = []
    for i in range(n - 1):
        P0 = ys_ext[i]
        P1 = ys_ext[i + 1]
        P2 = ys_ext[i + 2]
        P3 = ys_ext[i + 3]
        for j in range(points_per_segment):
            t = j / points_per_segment
            t2 = t * t
            t3 = t2 * t
            # Catmull-Rom 基本式 (パラメータ化なし)
            a = -0.5 * P0 + 1.5 * P1 - 1.5 * P2 + 0.5 * P3
            b = P0 - 2.5 * P1 + 2.0 * P2 - 0.5 * P3
            c = -0.5 * P0 + 0.5 * P2
            d = P1
            val = a * t3 + b * t2 + c * t + d
            xval = positions[i] + t * (positions[i + 1] - positions[i])
            xnew.append(xval)
            ynew.append(val)

    # 最後の点を追加
    xnew.append(positions[-1])
    ynew.append(ys[-1])
    return np.array(xnew), np.array(ynew)



def cal_relativescore(my,best) -> float:
    global maxormin
    #maxorminによって、相対スコアを変えるための関数
    if maxormin == "max":
        if best == 0:
            return 0.0 #正解者がいないということなので0にしておく
        return my/best
    if maxormin == "min":
        if my == 0:
            return 0.0
        return best/my


def cal_bestscore_each_seed(details:dict):
    """
    相対スコアおよびユニーク数を計算する関数。
    詳細:
    - bestscore: 各seed毎のbestスコア {seed:score}
    - comments_to_unique: コメント毎のユニーク(ベストスコアを獲得した)数 {comment:num}
    - comments_to_relative: コメント毎の相対スコア {comment:{seed:relative_score}}
    """
    # グローバル maxormin を参照して比較演算を決定
    global maxormin
    
    bestscore = {}  # 各seed毎のbestスコア {seed:score}
    comments_to_unique = defaultdict(int)  # コメント毎のユニーク数 {comment:num}
    comments_to_relative = {}  # コメント毎の相対スコア {comment:{seed:relative_score}}

    # 各seed毎にベストスコアを求める
    for seeds, scores, *_ in details.values():
        for i, seed in enumerate(seeds):
            now_score = bestscore.get(seed, -1)
            new_score = scores[i]
            if now_score == -1:
                # 初回設定
                bestscore[seed] = new_score
            else:
                # maxormin に応じて比較
                if maxormin == "max":
                    if new_score > now_score:
                        bestscore[seed] = new_score
                else:  # "min"
                    if new_score < now_score and new_score > 0:
                        bestscore[seed] = new_score
                    elif now_score == 0 and new_score > 0:
                        bestscore[seed] = new_score
    
    # ユニーク数と相対スコアを求める
    for seeds, scores, _, comment, *_ in details.values():
        seeds_to_relative_score = {}
        for i, seed in enumerate(seeds):
            best_score = bestscore[seed]
            myscore = scores[i]
            if scores[i] == best_score:
                # ベストスコアと一致する = そのseedにおいて最高スコアを獲得している
                comments_to_unique[comment] += 1
            relative_score = cal_relativescore(my=myscore, best=best_score)
            seeds_to_relative_score[seed] = relative_score
        
        comments_to_relative[comment] = seeds_to_relative_score
            
    return bestscore, comments_to_unique, comments_to_relative

#ここまで事前準備系関数
#---------------------------------------------------------------------
#ここから描画系関数

#ここまで描画系関数
#---------------------------------------------------------------------

def main():
    # Streamlit用ページ設定
    st.set_page_config(layout="wide", page_title="NCP Heuristic Visualizer")

    # configの読み込み (max/min と visualizer URL を取得)
    config = load_config()
    if not config or config.get("max_or_min") not in ["max","min"]:
        st.error("config.json が見つからないか、'max_or_min' の値が不正です。")
        return

    # グローバル変数の再初期化
    global maxormin
    maxormin = config.get("max_or_min")

    # logsフォルダから実行結果一覧を取得
    folders = fetch_data_from_logs(logs_dir)
    if folders is None:
        st.error("logs フォルダが見つからないか中身がありません。")
        return

    # 各フォルダから詳細データを作成
    details = fetch_data_from_floders(folders)
    if details is None:
        st.error("results.json の読み込みに失敗しました。フォルダ構成を確認してください。")
        return

    # 各seedごとのベストやユニーク数等を計算
    best_score,comments_to_unique,comments_to_relative = cal_bestscore_each_seed(details)

    # UI: 画面選択 (左サイドバーに配置)
    page = st.sidebar.selectbox("画面選択", ["Visualizer","Comparison"])

    # 実行結果一覧（新しい順）を作る。表示用の文字列と対応するPathを保持する。
    run_items = []  # (display_str, path)
    for path, vals in details.items():
        # details の 6 要素目までを安全に取り出す（7番目は shared_list）
        if len(vals) >= 6:
            seeds, scores, date, comment, sum_score, acstr = vals[:6]
        else:
            continue
        display = f"{comment} | {date} | {format_score_scientific(sum_score)} | {acstr}"
        run_items.append((display, path))

    # 新しい順ですでに並んでいるはずなのでそのまま使用

    if page == "Visualizer":
        # 左右分割: 左に実行結果一覧、右にビジュアライザー
        left, right = st.columns([1,2])

        with left:
            st.header("実行結果一覧")
            displays = [d for d,p in run_items]
            # ラジオで1つ選択するUIにする
            selected_display = st.radio("選択してください", displays)
            # 選ばれた項目のPathを取得
            selected_path = None
            for d,p in run_items:
                if d == selected_display:
                    selected_path = p
                    break

        # 右側でvisualizerを描画
        with right:
            st.header("ビジュアライザ")
            if selected_path is None:
                st.info("左から実行結果を選んでください。")
            else:
                vals = details[selected_path]
                if len(vals) >= 6:
                    seeds, scores, date, comment, sum_score, acstr = vals[:6]
                else:
                    st.error("選択した実行結果のデータ形式が不正です。")
                    return
                # seed選択: 昇順で表示。表示時に AC/WA を分かりやすく付与する
                seeds_sorted = sorted(seeds)
                display_options = []
                display_to_seed = {}
                for s in seeds_sorted:
                    try:
                        i = seeds.index(s)
                        status = "AC" if scores[i] > 0 else "WA"
                    except ValueError:
                        status = ""
                    disp = f"{s} ({status})" if status else str(s)
                    display_options.append(disp)
                    display_to_seed[disp] = s
                selected_disp = st.selectbox("seed を選択", display_options)
                selected_seed = display_to_seed[selected_disp]

                # 選ばれたseedに対応する out_<i>.txt を読み込む. results.jsonの順序に合わせるため元のインデックスを使用
                # 選択された seed のインデックス（results.json の順序に合わせる）
                try:
                    idx = seeds.index(selected_seed)
                except ValueError:
                    idx = None

                out_text = ""
                if idx is not None:
                    # out ファイルは 1-indexed (out_1.txt のように保存されている想定)
                    out_path = selected_path / f"out_{idx+1}.txt"
                    if out_path.exists():
                        try:
                            with out_path.open("r",encoding="utf-8") as f:
                                out_text = f.read()
                        except:
                            out_text = ""

                # 選択された out を所定のファイルに書き出す（visualizer がファイル参照する場合の補助）
                try:
                    current_out_path = selected_path / "current_out.txt"
                    with current_out_path.open("w", encoding="utf-8") as wf:
                        wf.write(out_text)
                    # 可搬性のためパスはスラッシュを使った文字列にする
                    current_out_path_str = str(current_out_path.as_posix())
                except Exception:
                    current_out_path_str = ""

                # visualizerのHTMLを取得 (configの visualizer_URL を優先、それが無ければローカルの visualize.html を使用)
                visualizer_html = None
                visualizer_url = config.get("visualizer_URL","")
                if visualizer_url:
                    visualizer_html = fetch_visualizer(visualizer_url)

                if visualizer_html is None:
                    # ローカルの visualize.html を読む
                    local_vis_path = Path(__file__).resolve().parent.parent / "visualize.html"
                    if local_vis_path.exists():
                        try:
                            with local_vis_path.open("r",encoding="utf-8") as f:
                                visualizer_html = f.read()
                        except:
                            visualizer_html = None

                if visualizer_html is None:
                    st.error("ビジュアライザーの取得に失敗しました (URL または visualize.html を確認)。")
                else:
                    # 初期 seed および out を visualizer 側で参照できるように前方スクリプトで渡す
                    # visualizer 側がこれらの変数を参照するように作られていることを期待する
                    # visualizer 側へ渡す前方スクリプト
                    # - window.initial_seed: 選択した seed の値
                    # - window.initial_out: 選択した out の中身（文字列）
                    # - window.initial_out_path: 書き出した所定ファイルのパス（存在する場合）
                    # visualizer 内の要素に直接値を入れて、自動で generateInput() と visualize() を呼ぶスクリプト
                    safe_seed = json.dumps(selected_seed)
                    safe_out = json.dumps(out_text)
                    safe_out_path = json.dumps(current_out_path_str)
                    # f-string 内に多数の中括弧が含まれていたため、ここでは文字列連結で安全に組み立てる
                    pre_script = (
                        "<script>(function(){const targetSeed = " + safe_seed
                        + "; const targetOut = " + safe_out
                        + "; const targetOutPath = " + safe_out_path
                        + "; function applyVals(){const s = document.getElementById('seedBox'); const o = document.getElementById('outputBox'); if(s) s.value = targetSeed; if(o) o.value = targetOut; try{ if(typeof generateInput === 'function') generateInput(); }catch(e){} try{ if(typeof visualize === 'function') visualize(); }catch(e){} } if(document.readyState==='loading'){document.addEventListener('DOMContentLoaded', applyVals);} else { setTimeout(applyVals,50); } let trials=0; const poll=setInterval(()=>{ if(document.getElementById('seedBox')){ applyVals(); clearInterval(poll);} else if(++trials>50){ clearInterval(poll);} } ,100); })();</script>"
                    )
                    # srcdoc的に埋め込む
                    full_html = pre_script + visualizer_html
                    # iframe 埋め込み
                    components.html(full_html, height=700, scrolling=True)

    else:
        # Comparison 画面
        st.header("実行結果比較")
        # 上: 実行結果一覧 (新しいもの上)
        st.subheader("実行結果一覧")
        # ページ上にチェックボックスを縦に並べ、ユーザーが直接選択できるようにする
        selected_multi = []
        for disp, pth in run_items:
            key = f"cmp_{pth.name}"
            if st.checkbox(disp, key=key):
                selected_multi.append(disp)

        # 下: 折れ線グラフ (seed:横, score:縦)
        st.subheader("スコア比較グラフ")
        if not selected_multi:
            st.info("上で1つ以上の実行結果を選択してください。")
        else:
            # まず、shared_vars に含まれるキーを収集し、横軸に使う値を選べるようにする
            shared_keys = set()
            for vals in details.values():
                if len(vals) > 6:
                    shared_list = vals[6]
                    for sv in shared_list:
                        if isinstance(sv, dict):
                            shared_keys.update(sv.keys())

            x_options = ["seed"] + sorted(shared_keys)
            x_choice = st.selectbox("横軸に使う値", x_options)

            # 選択中のすべての実行結果から選択された横軸の全値集合を作る
            all_x_values = set()
            for d in selected_multi:
                p = None
                for disp,pp in run_items:
                    if disp == d:
                        p = pp
                        break
                if p is None:
                    continue
                vals = details[p]
                seeds = vals[0]
                scores = vals[1]
                shared_list = vals[6] if len(vals) > 6 else []
                for i, sd in enumerate(seeds):
                    if x_choice == "seed":
                        xval = sd
                    else:
                        if i < len(shared_list) and isinstance(shared_list[i], dict):
                            xval = shared_list[i].get(x_choice, None)
                        else:
                            xval = None
                    try:
                        if xval is None:
                            continue
                        xnum = float(xval)
                    except Exception:
                        continue
                    all_x_values.add(xnum)

            all_x_values = sorted(all_x_values)

            # y軸を対数スケールにするかどうかの設定
            log_scale = st.checkbox("y 軸を対数スケールにする", value=False)

            # まず全実行を走査して、選択した横軸ごとの平均スコアを算出して集計する
            run_aggregates = []  # list of dict: {comment, x_unique(list), y_mean(list)}
            from collections import defaultdict as _dd
            union_x = set()
            for d in selected_multi:
                p = None
                for disp,pp in run_items:
                    if disp == d:
                        p = pp
                        break
                if p is None:
                    continue
                vals = details[p]
                seeds = vals[0]
                scores = vals[1]
                comment = vals[3]
                shared_list = vals[6] if len(vals) > 6 else []

                _agg = _dd(list)
                for i, sc in enumerate(scores):
                    if x_choice == "seed":
                        xval = seeds[i]
                    else:
                        if i < len(shared_list) and isinstance(shared_list[i], dict):
                            xval = shared_list[i].get(x_choice, None)
                        else:
                            xval = None
                    if xval is None:
                        continue
                    try:
                        xnum = float(xval)
                    except Exception:
                        continue
                    _agg[xnum].append(sc)

                x_unique = sorted(_agg.keys())
                y_mean = [sum(_agg[x]) / len(_agg[x]) for x in x_unique]
                union_x.update(x_unique)
                run_aggregates.append({"comment": comment, "x_unique": x_unique, "y_mean": y_mean})

            # 各 x に対するベストスコア（全 runs を対象に計算する）
            best_by_x = {}
            # 全 runs を走査して、各 run ごとに x->mean を算出し、その中で best を決める
            for vals in details.values():
                seeds_all = vals[0]
                scores_all = vals[1]
                shared_list_all = vals[6] if len(vals) > 6 else []
                tmp = defaultdict(list)
                for i, sc in enumerate(scores_all):
                    if x_choice == "seed":
                        xval = seeds_all[i]
                    else:
                        if i < len(shared_list_all) and isinstance(shared_list_all[i], dict):
                            xval = shared_list_all[i].get(x_choice, None)
                        else:
                            xval = None
                    if xval is None:
                        continue
                    try:
                        xnum = float(xval)
                    except Exception:
                        continue
                    tmp[xnum].append(sc)

                for xk, lst in tmp.items():
                    mean_val = sum(lst) / len(lst)
                    if xk in best_by_x:
                        # maxormin グローバル変数に応じて max または min で比較する
                        if maxormin == "max":
                            best_by_x[xk] = max(best_by_x[xk], mean_val)
                        else:  # "min"
                            best_by_x[xk] = min(best_by_x[xk], mean_val)
                    else:
                        best_by_x[xk] = mean_val

            # 描画フェーズ
            fig_abs = go.Figure()
            fig_rel = go.Figure()
            for r in run_aggregates:
                comment = r["comment"]
                x_unique = r["x_unique"]
                y_mean = r["y_mean"]

                # 対数スケール時に 0 や負の値があると描画エラーになるための安全化
                def _sanitize_for_log(vals):
                    arr = np.array(vals, dtype=float)
                    pos = arr[arr > 0]
                    if pos.size > 0:
                        eps = float(pos.min()) * 0.1
                        if eps <= 0:
                            eps = 1e-6
                    else:
                        eps = 1e-6
                    arr[arr <= 0] = eps
                    return arr

                if len(x_unique) == 1:
                    if log_scale:
                        y_plot = float(_sanitize_for_log([y_mean[0]])[0])
                    else:
                        y_plot = float(y_mean[0])
                    fig_abs.add_trace(go.Scatter(x=[0], y=[y_plot], mode='markers', marker=dict(size=8), name=comment,
                                                 customdata=[x_unique[0]],
                                                 hovertemplate=f'{comment}<br>{x_choice}: %{{customdata:.0f}}<br>score: %{{y:.0f}}<extra></extra>'))

                    # 相対値は best_by_x を使って計算
                    best_val = best_by_x.get(x_unique[0], None)
                    if best_val is None:
                        rel_plot = 0.0
                    else:
                        rel_plot = cal_relativescore(y_mean[0], best_val)
                    if log_scale:
                        rel_plot = float(_sanitize_for_log([rel_plot])[0])
                    fig_rel.add_trace(go.Scatter(x=[0], y=[rel_plot], mode='markers', marker=dict(size=8), name=comment,
                                                 customdata=[x_unique[0]],
                                                 hovertemplate=f'{comment}<br>{x_choice}: %{{customdata:.0f}}<br>relative: %{{y:.5f}}<extra></extra>'))
                else:
                    positions = list(range(len(x_unique)))
                    xnew_pos, ynew = catmull_rom_chain(positions, y_mean, points_per_segment=300)
                    if log_scale:
                        ynew_plot = _sanitize_for_log(ynew)
                        y_mean_plot = _sanitize_for_log(y_mean)
                    else:
                        ynew_plot = np.array(ynew, dtype=float)
                        y_mean_plot = np.array(y_mean, dtype=float)

                    if x_choice == "seed":
                        fig_abs.add_trace(go.Scatter(x=xnew_pos, y=ynew_plot, mode='lines', name=comment,
                                                     line=dict(width=2, shape='spline', smoothing=1.3), hoverinfo='skip'))
                        fig_abs.add_trace(go.Scatter(x=positions, y=y_mean_plot, mode='markers', marker=dict(size=8),
                                                     name=f"{comment} (points)", customdata=x_unique,
                                                     hovertemplate='seed: %{customdata:.0f}<br>score: %{y:.0f}<extra></extra>'))
                    else:
                        xnew_global = np.interp(xnew_pos, positions, x_unique)
                        fig_abs.add_trace(go.Scatter(x=xnew_global, y=ynew_plot, mode='lines', name=comment,
                                                     line=dict(width=2, shape='spline', smoothing=1.3), hoverinfo='skip'))
                        fig_abs.add_trace(go.Scatter(x=x_unique, y=y_mean_plot, mode='markers', marker=dict(size=8),
                                                     name=f"{comment} (points)", customdata=x_unique,
                                                     hovertemplate=f'{x_choice}: %{{customdata:.0f}}<br>score: %{{y:.0f}}<extra></extra>'))

                    # 相対値: 各 x に対して best_by_x を参照して相対スコアを算出
                    y_rel = []
                    for xi, ym in zip(x_unique, y_mean):
                        best_val = best_by_x.get(xi, None)
                        if best_val is None:
                            y_rel.append(0.0)
                        else:
                            y_rel.append(cal_relativescore(ym, best_val))
                    xnew_rel, ynew_rel = catmull_rom_chain(positions, y_rel, points_per_segment=300)
                    if log_scale:
                        ynew_rel_plot = _sanitize_for_log(ynew_rel)
                        y_rel_plot = _sanitize_for_log(y_rel)
                    else:
                        ynew_rel_plot = np.array(ynew_rel, dtype=float)
                        y_rel_plot = np.array(y_rel, dtype=float)

                    if x_choice == "seed":
                        fig_rel.add_trace(go.Scatter(x=xnew_rel, y=ynew_rel_plot, mode='lines', name=comment,
                                                     line=dict(width=2, shape='spline', smoothing=1.3), hoverinfo='skip'))
                        fig_rel.add_trace(go.Scatter(x=positions, y=y_rel_plot, mode='markers', marker=dict(size=8),
                                                     name=f"{comment} (points)", customdata=x_unique,
                                                     hovertemplate='seed: %{customdata:.0f}<br>relative: %{y:.5f}<extra></extra>'))
                    else:
                        xnew_global_rel = np.interp(xnew_rel, positions, x_unique)
                        fig_rel.add_trace(go.Scatter(x=xnew_global_rel, y=ynew_rel_plot, mode='lines', name=comment,
                                                     line=dict(width=2, shape='spline', smoothing=1.3), hoverinfo='skip'))
                        fig_rel.add_trace(go.Scatter(x=x_unique, y=y_rel_plot, mode='markers', marker=dict(size=8),
                                                     name=f"{comment} (points)", customdata=x_unique,
                                                     hovertemplate=f'{x_choice}: %{{customdata:.0f}}<br>relative: %{{y:.5f}}<extra></extra>'))

            # x 軸の表示設定: seed の場合は等間隔インデックス扱い、それ以外は実数値で表示
            if x_choice == "seed":
                fig_abs.update_xaxes(showticklabels=False, range=[-0.5, max(0,len(all_x_values)-0.5)])
                fig_rel.update_xaxes(showticklabels=False, range=[-0.5, max(0,len(all_x_values)-0.5)])
            else:
                fig_abs.update_xaxes(showticklabels=True, autorange=True)
                fig_rel.update_xaxes(showticklabels=True, autorange=True)
            # レイアウト調整
            y_axis_type = 'log' if log_scale else 'linear'
            fig_abs.update_layout(yaxis_title='score', hovermode='x unified', height=420, width=1200,
                              shapes=[dict(type="rect", xref="paper", yref="paper",
                                           x0=0, y0=0, x1=1, y1=1,
                                           line=dict(color="black", width=1), fillcolor="rgba(0,0,0,0)")],
                              margin=dict(l=20, r=20, t=20, b=20),
                              transition=dict(duration=600, easing='cubic-in-out'),
                              autosize=True,
                              yaxis_type=y_axis_type)
            fig_rel.update_layout(yaxis_title='relative score', hovermode='x unified', height=420, width=1200,
                              shapes=[dict(type="rect", xref="paper", yref="paper",
                                           x0=0, y0=0, x1=1, y1=1,
                                           line=dict(color="black", width=1), fillcolor="rgba(0,0,0,0)")],
                              margin=dict(l=20, r=20, t=20, b=20),
                              transition=dict(duration=600, easing='cubic-in-out'),
                              autosize=True,
                              yaxis_type=y_axis_type)

            # 表示モードを切替 (絶対スコア / 相対スコア)
            plot_mode = st.radio("表示モード", ("Absolute score","Relative score"), index=0, horizontal=True)
            if plot_mode == "Absolute score":
                st.plotly_chart(fig_abs)
            else:
                st.plotly_chart(fig_rel)

            # 選択した横軸値の詳細を縦に表示する機能
            if all_x_values:
                values_list = sorted(all_x_values)
                inspect_x = st.selectbox(f"詳細を表示する {x_choice} の値を選択", values_list)
                st.markdown(f"**{x_choice} = {inspect_x}**")
                # 各選択実行についてコメント: score を縦に表示（選択した横軸値に対する平均値）
                for d in selected_multi:
                    p = None
                    for disp,pp in run_items:
                        if disp == d:
                            p = pp
                            break
                    if p is None:
                        continue
                    vals = details[p]
                    seeds = vals[0]
                    scores = vals[1]
                    comment = vals[3]
                    shared_list = vals[6] if len(vals) > 6 else []

                    # 横軸選択に応じて当該値に該当するスコアを集め平均化
                    matched = []
                    matched_rel = []
                    for i, sc in enumerate(scores):
                        if x_choice == "seed":
                            xval = seeds[i]
                        else:
                            if i < len(shared_list) and isinstance(shared_list[i], dict):
                                xval = shared_list[i].get(x_choice, None)
                            else:
                                xval = None
                        try:
                            if xval is None:
                                continue
                            xnum = float(xval)
                        except Exception:
                            continue
                        if abs(xnum - float(inspect_x)) < 1e-9:
                            matched.append(sc)
                            # 相対値もあれば収集
                            rel_map = comments_to_relative.get(comment, {})
                            # rel_map keys are seeds -> relative score
                            seed_val = seeds[i]
                            if seed_val in rel_map:
                                matched_rel.append(rel_map.get(seed_val, None))

                    if matched:
                        score_val = sum(matched) / len(matched)
                    else:
                        score_val = None

                    if plot_mode == "Absolute score":
                        display_val = f"{score_val:.0f}" if score_val is not None else "-"
                    else:
                        if matched_rel:
                            rel_v = sum(matched_rel) / len(matched_rel)
                            display_val = f"{rel_v:.5f}"
                        else:
                            display_val = "-"

                    st.write(f"- {comment} : {display_val}")


if __name__ == '__main__':
    main()
