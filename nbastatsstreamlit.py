from nba_api.stats.static import players
from nba_api.stats.endpoints import playercareerstats, commonplayerinfo
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime
from streamlit.runtime.scriptrunner import RerunException
from streamlit.runtime.scriptrunner import get_script_run_ctx
import os
import google.generativeai as genai
import numpy as np

GEMINI_API_KEY = "AIzaSyC0APdZnTD2G3GtPUBmWwB-XhNauX4TqUo"
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("models/gemini-2.5-flash-lite")


def generate_player_summary(player_name, stats_df, advanced_df):
    if stats_df.empty:
        return f"No available stats for {player_name}."

    summary = f"📊 Full season-by-season stats for **{player_name}**:\n\n"

    # Loop through each season
    for i in range(len(stats_df)):
        season_stats = stats_df.iloc[i]
        season_adv = advanced_df.iloc[i]

        summary += f"---\n"
        summary += f"### Season {season_stats['SEASON_ID']} ({season_stats['TEAM_ABBREVIATION']})\n"
        summary += f"- **PPG:** {season_stats['PTS']:.1f}, **RPG:** {season_stats['REB']:.1f}, **APG:** {season_stats['AST']:.1f}\n"
        summary += f"- **SPG:** {season_stats['STL']:.1f}, **BPG:** {season_stats['BLK']:.1f}, **TPG:** {season_stats['TOV']:.1f}\n"
        summary += f"- **Games Played:** {season_stats['GP']}, **Minutes/Game:** {season_stats['MIN']:.1f}\n"

        # Shooting splits
        if 'FG%' in season_stats:
            summary += f"- **FG%:** {season_stats['FG%']:.1f}%, **3P%:** {season_stats['3P%']:.1f}%, **FT%:** {season_stats['FT%']:.1f}%\n"

        # Advanced shooting efficiency
        summary += f"- **TS%:** {season_adv.get('TS%', 0):.2f}%, **EFG%:** {season_adv.get('EFG%', 0):.2f}%, **PPS:** {season_adv.get('PPS', 0):.2f}\n"
        summary += f"- **USG% (est):** {season_adv.get('USG% (est)', 0):.2f}%\n"

        # Per-36 numbers
        summary += (
            f"- **PTS/36:** {season_adv.get('PTS/36', 0):.2f}, "
            f"**REB/36:** {season_adv.get('REB/36', 0):.2f}, "
            f"**AST/36:** {season_adv.get('AST/36', 0):.2f}, "
            f"**STL/36:** {season_adv.get('STL/36', 0):.2f}, "
            f"**BLK/36:** {season_adv.get('BLK/36', 0):.2f}, "
            f"**TOV/36:** {season_adv.get('TOV/36', 0):.2f}\n"
        )
        summary += f"- **AST/TO Ratio:** {season_adv.get('AST/TO', 0):.2f}\n\n"

    return summary



# Correct USG% (Usage Rate) formula for your Streamlit NBA app
# This formula is more realistic and replicates the NBA advanced stats logic as closely as possible



def compute_full_advanced_stats(df):
    df = df.copy()

    # Avoid division by zero
    df['MIN'] = df['MIN'].replace(0, None)
    df['GP'] = df['GP'].replace(0, None)
    df['FGA'] = df['FGA'].replace(0, None)

    # Traditional percentages (multiply to get % format)
    if 'FG_PCT' in df.columns: df['FG%'] = df['FG_PCT'] * 100
    if 'FG3_PCT' in df.columns: df['3P%'] = df['FG3_PCT'] * 100
    if 'FT_PCT' in df.columns: df['FT%'] = df['FT_PCT'] * 100

    # Advanced shooting efficiency
    df['TS%'] = df.apply(lambda row: (row['PTS'] / (2 * (row['FGA'] + 0.44 * row['FTA'])) * 100)
                        if (row['FGA'] + 0.44 * row['FTA']) > 0 else None, axis=1)

    df['EFG%'] = df.apply(lambda row: (row['FGM'] + 0.5 * row['FG3M']) / row['FGA'] * 100
                          if row['FGA'] else None, axis=1)

    # Assist to Turnover Ratio
    df['AST/TO'] = df.apply(lambda row: row['AST'] / row['TOV']
                            if row['TOV'] else None, axis=1)

    # Points Per Shot
    df['PPS'] = df.apply(lambda row: row['PTS'] / row['FGA']
                         if row['FGA'] else None, axis=1)

    # PER (very rough estimate, not official)
    df['PER'] = df.apply(lambda row:
        (row['PTS'] + row['REB'] + row['AST'] + row['STL'] + row['BLK']
         - (row['FGA'] - row['FGM']) - (row['FTA'] - row['FTM']) - row['TOV']) / row['GP']
        if row['GP'] else None, axis=1)

    # Per-36 minute stats
    df['PTS/36'] = (df['PTS'] / df['MIN']) * 36
    df['REB/36'] = (df['REB'] / df['MIN']) * 36
    df['AST/36'] = (df['AST'] / df['MIN']) * 36
    df['STL/36'] = (df['STL'] / df['MIN']) * 36
    df['BLK/36'] = (df['BLK'] / df['MIN']) * 36
    df['TOV/36'] = (df['TOV'] / df['MIN']) * 36
    df['FGM/36'] = (df['FGM'] / df['MIN']) * 36
    df['FGA/36'] = (df['FGA'] / df['MIN']) * 36
    df['FG3M/36'] = (df['FG3M'] / df['MIN']) * 36
    df['OREB/36'] = (df['OREB'] / df['MIN']) * 36
    df['DREB/36'] = (df['DREB'] / df['MIN']) * 36

    # USG% Placeholder: better value should come from NBA.com advanced endpoint
    df['USG% (est)'] = df.apply(lambda row:
        100 * (row['FGA'] + 0.44 * row['FTA'] + row['TOV']) / row['MIN']
        if row['MIN'] else None, axis=1)

    # Round all % or float columns for display
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = df[col].round(2)

    # Clean up unwanted ID fields
    df = df.drop(columns=[col for col in ['PLAYER_ID', 'TEAM_ID', 'LEAGUE_ID'] if col in df.columns], errors='ignore')

    return df



     
            


# Your college_logos dictionary remains the same...

st.title("NBA Advanced Player Stats Explorer")
college_logos = {
    "Duke": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/Duke_Blue_Devils_basketball_mark.svg/300px-Duke_Blue_Devils_basketball_mark.svg.png",
    "North Carolina": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d7/North_Carolina_Tar_Heels_logo.svg/500px-North_Carolina_Tar_Heels_logo.svg.png",
    "Kentucky": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Kentucky_Wildcats_logo.svg/300px-Kentucky_Wildcats_logo.svg.png",
    "Kansas": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/90/Kansas_Jayhawks_1946_logo.svg/400px-Kansas_Jayhawks_1946_logo.svg.png",
    "UCLA": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d1/UCLA_Bruins_primary_logo.svg/400px-UCLA_Bruins_primary_logo.svg.png",
    "Ohio State": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c1/Ohio_State_Buckeyes_logo.svg/400px-Ohio_State_Buckeyes_logo.svg.png",
    "Texas": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/Texas_Longhorns_logo.svg/450px-Texas_Longhorns_logo.svg.png",
    "Southern California": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/94/USC_Trojans_logo.svg/244px-USC_Trojans_logo.svg.png",
    "Michigan": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fb/Michigan_Wolverines_logo.svg/300px-Michigan_Wolverines_logo.svg.png",
    "Arizona": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/Arizona_Wildcats_logo.svg/300px-Arizona_Wildcats_logo.svg.png",
    "Arkansas": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Arkansas_wordmark_2014.png/500px-Arkansas_wordmark_2014.png",
    "Metropolitans 92": "https://upload.wikimedia.org/wikipedia/en/2/2a/Metropolitans_92_logo.png",
    "NBA G League Ignite": "https://upload.wikimedia.org/wikipedia/en/thumb/8/88/NBA_G_League_Ignite_logo_%282022%29.svg/400px-NBA_G_League_Ignite_logo_%282022%29.svg.png",
    "St. Vincent-St. Mary HS (OH)": "https://upload.wikimedia.org/wikipedia/en/b/be/St._Vincent-St._Mary_High_School_logo.png",
    "Murray State": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/df/Murray_State_Racers_wordmark.svg/500px-Murray_State_Racers_wordmark.svg.png",
    "Oklahoma": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/61/Oklahoma_Sooners_logo.svg/250px-Oklahoma_Sooners_logo.svg.png",
    "Oklahoma State": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/01/Oklahoma_State_University_system_logo.svg/450px-Oklahoma_State_University_system_logo.svg.png",
    "Iowa State": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f9/Iowa_State_Cyclones_logo.svg/300px-Iowa_State_Cyclones_logo.svg.png",
    "Florida": "https://upload.wikimedia.org/wikipedia/en/thumb/9/99/Florida_Gators_men%27s_basketball_logo.svg/400px-Florida_Gators_men%27s_basketball_logo.svg.png",
    "Florida State": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/20/Florida_State_Athletics_wordmark.svg/500px-Florida_State_Athletics_wordmark.svg.png",
    "Stanford": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4b/Stanford_Cardinal_logo.svg/200px-Stanford_Cardinal_logo.svg.png",
    "Indiana": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/Indiana_Hoosiers_logo.svg/250px-Indiana_Hoosiers_logo.svg.png",
    "TCU": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/TCU_Horned_Frogs_logo.svg/350px-TCU_Horned_Frogs_logo.svg.png",
    "Louisiana State": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4a/LSU_Athletics_logo.svg/400px-LSU_Athletics_logo.svg.png",
    "Oregon": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/Oregon_Ducks_logo.svg/250px-Oregon_Ducks_logo.svg.png",
    "Auburn": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Auburn_Tigers_logo.svg/300px-Auburn_Tigers_logo.svg.png",
    "Vanderbilt": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/Vanderbilt_Athletics_logo.svg/330px-Vanderbilt_Athletics_logo.svg.png",
    "Villanova": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/23/Villanova_Wildcats_logo.svg/300px-Villanova_Wildcats_logo.svg.png",
    "Real Madrid": "https://upload.wikimedia.org/wikipedia/en/b/be/Real_Madrid_Baloncesto.png",
    "Alabama": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Alabama_Crimson_Tide_logo.svg/300px-Alabama_Crimson_Tide_logo.svg.png",
    "FC Barcelona": "https://upload.wikimedia.org/wikipedia/en/thumb/4/47/FC_Barcelona_%28crest%29.svg/410px-FC_Barcelona_%28crest%29.svg.png",
    "Conneticut": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b3/Connecticut_Huskies_wordmark.svg/500px-Connecticut_Huskies_wordmark.svg.png",
    "Marquette": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/Marquette_Golden_Eagles_logo.svg/300px-Marquette_Golden_Eagles_logo.svg.png",
    "Syracuse": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/49/Syracuse_Orange_logo.svg/200px-Syracuse_Orange_logo.svg.png",
    "Baylor": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c4/Baylor_Athletics_logo.svg/300px-Baylor_Athletics_logo.svg.png",
    "Creighton": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/54/Creighton_athletics_wordmark_2013.png/500px-Creighton_athletics_wordmark_2013.png",
    "Purdue": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/35/Purdue_Boilermakers_logo.svg/300px-Purdue_Boilermakers_logo.svg.png",
    "Memphis": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/06/Memphis_Tigers_primary_wordmark.svg/500px-Memphis_Tigers_primary_wordmark.svg.png",
    "Illinois": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Illinois_Fighting_Illini_logo.svg/200px-Illinois_Fighting_Illini_logo.svg.png",
    "Seton Hall": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Seton_Hall_Pirates_wordmark.svg/250px-Seton_Hall_Pirates_wordmark.svg.png",
    "Louisville": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/df/Louisville_Wordmark_%282023%29.svg/500px-Louisville_Wordmark_%282023%29.svg.png",
    "Michigan State": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/df/Michigan_State_Spartans_wordmark.svg/400px-Michigan_State_Spartans_wordmark.svg.png",
    "Wake Forest": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Wake_Forest_University_Athletic_logo.svg/300px-Wake_Forest_University_Athletic_logo.svg.png",
    "Notre Dame": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/Nd_athletics_gold_logo_2015.svg/300px-Nd_athletics_gold_logo_2015.svg.png",
    "Georgia": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/80/Georgia_Athletics_logo.svg/400px-Georgia_Athletics_logo.svg.png",
    "Missouri": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/Mizzou_Athletics_wordmark.svg/500px-Mizzou_Athletics_wordmark.svg.png",
    "Georgetown": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Georgetown_Hoyas_logo.svg/300px-Georgetown_Hoyas_logo.svg.png",
    "Texas A&M": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ee/Texas_A%26M_University_logo.svg/300px-Texas_A%26M_University_logo.svg.png",
    "NBA Global Academy": "https://cdn.nba.com/logos/microsites/nba-academy.svg",
    "Gonzaga": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bf/Gonzaga_Bulldogs_wordmark.svg/450px-Gonzaga_Bulldogs_wordmark.svg.png",
    "Iowa": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/01/Iowa_Hawkeyes_wordmark.svg/400px-Iowa_Hawkeyes_wordmark.svg.png",
    "Providence": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/Providence_wordmark1_2002.png/500px-Providence_wordmark1_2002.png",
    "Georgia Tech": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bf/Georgia_Tech_Yellow_Jackets_logo.svg/350px-Georgia_Tech_Yellow_Jackets_logo.svg.png",
    "Maryland": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a6/Maryland_Terrapins_logo.svg/250px-Maryland_Terrapins_logo.svg.png",
    "Wisconsin": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1d/Wisconsin_Badgers_logo_basketball_red.svg/400px-Wisconsin_Badgers_logo_basketball_red.svg.png",
    "UNLV": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f6/UNLV_Rebels_wordmark.svg/500px-UNLV_Rebels_wordmark.svg.png",
    "Brigham Young": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/BYU_Stretch_Y_Logo.png/500px-BYU_Stretch_Y_Logo.png",
    "Houston": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/Houston_Cougars_primary_logo.svg/300px-Houston_Cougars_primary_logo.svg.png",
    "Colorado": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/20/Colorado_Buffaloes_wordmark_black.svg/400px-Colorado_Buffaloes_wordmark_black.svg.png",
    "Washington": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/17/Washington_Huskies_logo.svg/300px-Washington_Huskies_logo.svg.png",
    "Utah": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/53/Utah_Utes_primary_logo.svg/300px-Utah_Utes_primary_logo.svg.png",
    "Boise State": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Boise_State_Broncos_wordmark.svg/500px-Boise_State_Broncos_wordmark.svg.png",
    "Xavier": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/Xavier_wordmark-basketball-fc-lt.svg/400px-Xavier_wordmark-basketball-fc-lt.svg.png",
    "California": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8b/California_Golden_Bears_logo.svg/250px-California_Golden_Bears_logo.svg.png",
    "Arizona State": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6c/Arizona_State_Athletics_wordmark.svg/500px-Arizona_State_Athletics_wordmark.svg.png",
    "Pittsburgh": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/Pitt_Panthers_wordmark.svg/350px-Pitt_Panthers_wordmark.svg.png",
    "St John's": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/63/St_johns_wordmark_red_2015.png/500px-St_johns_wordmark_red_2015.png",
    "Davidson": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/Davidson_Wildcats_logo.png/250px-Davidson_Wildcats_logo.png",
}

name = st.text_input("Enter an NBA player's name:")

# Reset session state if new name typed
if name and 'last_name' in st.session_state and name != st.session_state['last_name']:
    st.session_state.pop('player', None)
    st.session_state.pop('matches', None)
    st.session_state.pop('selected_index', None)

if st.button("Search"):
    matches = players.find_players_by_full_name(name)
    st.session_state['matches'] = matches 
    exact_matches = [p for p in matches if p['full_name'].lower() == name.lower()]
    matches = exact_matches if exact_matches else matches

    if matches:
        if len(matches) > 1:
            st.session_state['matches'] = matches  # Save matches for dropdown
        else:
            st.session_state['player'] = matches[0]
            st.session_state['matches'] = []
    else: 
        st.markdown("<span style='color:red;'>❌ No players found. Please check the spelling or try a different name.</span>", unsafe_allow_html=True)

if 'matches' in st.session_state and st.session_state['matches']:
    matches = st.session_state['matches']
    st.write("Multiple players found with that name:")
    options = {f"{p['full_name']} (ID: {p['id']})": p for p in matches}
    selected = st.radio("Select a player:", list(options.keys()), index=None, key="player_selection_radio")

    if selected:
        st.session_state['player'] = options[selected]
        st.session_state['matches'] = []  # Clear matches after selection
        st.rerun()


if 'player' in st.session_state:
    player = st.session_state['player']
    st.write(f"You selected: {player['full_name']} (ID: {player['id']})")
    info = commonplayerinfo.CommonPlayerInfo(player_id=player['id']).get_data_frames()[0]

    birthdate_str = info.loc[0, 'BIRTHDATE']
    birthdate = datetime.strptime(birthdate_str.split('T')[0], '%Y-%m-%d')
    today = datetime.today()
    age = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
    height = info.loc[0, 'HEIGHT']
    weight = info.loc[0, 'WEIGHT']
    position = info.loc[0, 'POSITION']
    college = info.loc[0, 'SCHOOL']
    teamID = info.loc[0, 'TEAM_ID']
    nbateam = info.loc[0, 'TEAM_NAME']
    headshot_url = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player['id']}.png"
    team_logo_url = f"https://cdn.nba.com/logos/nba/{teamID}/global/L/logo.svg"

    st.subheader("Player Info")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(headshot_url, caption=player['full_name'], width=200)
    with col2:
        st.markdown(f"**Age:** {age}")
        st.markdown(f"**Height:** {height}")
        st.markdown(f"**Weight:** {weight} lbs")
        st.markdown(f"**Position:** {position}")
        st.markdown(f"**College:** {college}")
        if college:
            college_logo_url = college_logos.get(college)
            col_logo1, col_logo2 = st.columns([1, 5])
            with col_logo1:
                if college_logo_url:
                    st.image(college_logo_url, width=100)
                else:
                    st.write("No logo available")
            with col_logo2:
                st.image(team_logo_url, width=100)

    choice = st.radio("What do you want to do?", ["See Stats", "Compare Players"])

    if choice == "See Stats":
        career = playercareerstats.PlayerCareerStats(player_id=player['id'], per_mode36='PerGame')
        stats = career.get_data_frames()[0]

        if stats.empty:
            st.write("No stats available.")
        else:
            # Copy to avoid messing with original stats
            advanced_stats = stats.copy()

            # Advanced calculations
            advanced_stats['TS%'] = advanced_stats.apply(
                lambda row: (row['PTS'] / (2 * (row['FGA'] + 0.44 * row['FTA'])) * 100)
                if (row['FGA'] + 0.44 * row['FTA']) > 0 else None, axis=1)

            advanced_stats['EFG%'] = advanced_stats.apply(
                lambda row: ((row['FGM'] + 0.5 * row['FG3M']) / row['FGA'] * 100)
                if row['FGA'] > 0 else None, axis=1)

            advanced_stats['PER'] = advanced_stats.apply(
                lambda row: round((row['PTS'] + row['REB'] + row['AST'] + row['STL'] + row['BLK']
                                - (row['FGA'] - row['FGM']) - (row['FTA'] - row['FTM']) - row['TOV']) / row['GP'], 2)
                if row['GP'] > 0 else None, axis=1)

            advanced_stats['AST/TO'] = advanced_stats.apply(
                lambda row: round(row['AST'] / row['TOV'], 2) if row['TOV'] > 0 else None, axis=1)

            advanced_stats['USG% (est)'] = advanced_stats.apply(
                lambda row: round(100 * ((row['FGA'] + 0.44 * row['FTA'] + row['TOV']) / (row['MIN'] + 1e-5)), 2)
                if row['MIN'] > 0 else None, axis=1)

            # Per-36 Minute Stats
            per_36_stats = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FGM', 'FGA', 'FG3M', 'OREB', 'DREB']
            for stat in per_36_stats:
                col_name = f"{stat}/36"
                advanced_stats[col_name] = advanced_stats.apply(
                    lambda row: round((row[stat] / row['MIN']) * 36, 2) if row['MIN'] > 0 else None, axis=1)

            # Shooting Percentages
            advanced_stats['FG%'] = advanced_stats['FG_PCT'] * 100
            advanced_stats['3P%'] = advanced_stats['FG3_PCT'] * 100
            advanced_stats['FT%'] = advanced_stats['FT_PCT'] * 100
            advanced_stats['PPS'] = advanced_stats.apply(
                lambda row: round(row['PTS'] / row['FGA'], 2) if row['FGA'] > 0 else None, axis=1)

            # Round % columns
            percentage_cols = ['TS%', 'EFG%', 'FG%', '3P%', 'FT%']
            for col in percentage_cols:
                if col in advanced_stats.columns:
                    advanced_stats[col] = advanced_stats[col].round(2)

            # Drop raw API %s
            advanced_stats = advanced_stats.drop(columns=['FG_PCT', 'FG3_PCT', 'FT_PCT'], errors='ignore')


            # Columns to display
            display_cols = [
                'SEASON_ID', 'TEAM_ABBREVIATION', 'GP', 'MIN',

                # Base stats
                'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV',
                'FGA', 'FGM', 'FG%', 'FG3A', 'FG3M', '3P%',
                'FTA', 'FTM', 'FT%',

                # Advanced shooting/efficiency
                'TS%', 'EFG%', 'PPS', 'PER', 'AST/TO', 'USG% (est)',

                # Per-36 Minute stats
                'PTS/36', 'REB/36', 'AST/36', 'STL/36', 'BLK/36', 'TOV/36',
                'FGM/36', 'FGA/36', 'FG3M/36', 'OREB/36', 'DREB/36'
            ]

            # Only keep columns that actually exist in the dataframe
            display_cols = [col for col in display_cols if col in advanced_stats.columns]

            st.dataframe(advanced_stats[display_cols])
            with st.expander("🧠 Ask the AI Assistant about this player"):
                user_question = st.text_input("Ask something about this player:", key="ai_question")
                if user_question:
                    # Generate summary
                    player_summary = generate_player_summary(player['full_name'], advanced_stats, advanced_stats)

                    # Build full prompt
                    full_prompt = f"You are an expert NBA analyst. Here is the stat summary for {player['full_name']}:\n\n{player_summary}\n\nQuestion: {user_question}\n\nTake the USG% and Per with a grain of salt. Grab it from a credible source if possible.Answer like you're explaining to a smart basketball fan:"

                    # Call Gemini
                    with st.spinner("Analyzing..."):
                        response = model.generate_content(full_prompt, generation_config={"max_output_tokens": 2048, "temperature": 0.7})
                        st.markdown("### 🧠 AI Analysis")
                        st.write(response.text)


        
        if stats.empty:
            st.write("No stats available.")
    elif choice == "Compare Players":
        other_name = st.text_input("Enter another player's name to compare:")

        if other_name:
            other_matches = players.find_players_by_full_name(other_name)
            exact_matches = [p for p in other_matches if p['full_name'].lower() == other_name.lower()]
            other_matches = exact_matches if exact_matches else other_matches
            if "other_matches" not in st.session_state:
                st.session_state['other_matches'] = []

            if other_matches:
                st.session_state['other_matches'] = other_matches
                if len(other_matches) > 1:
                    st.write("Multiple players found with that name:")

                    options2 = {f"{p['full_name']} (ID: {p['id']})": p for p in other_matches}

                    other_selected = st.radio("Select a player:",list(options2.keys()),index = None, key="other_player_selection_radio" )

                    if other_selected:
                        st.session_state['other_player'] = options2[other_selected]
                        st.session_state['other_matches'] = []  # Clear matches after selection
                else:
                    st.session_state['other_player'] = other_matches[0]

                other_player = st.session_state.get('other_player', None)

                if other_player:
                    st.write(f"You selected: {other_player['full_name']}")
                    info2 = commonplayerinfo.CommonPlayerInfo(player_id=other_player['id']).get_data_frames()[0]

                    birthdate_str2 = info2.loc[0, 'BIRTHDATE']
                    birthdate2 = datetime.strptime(birthdate_str2.split('T')[0], '%Y-%m-%d')
                    today2 = datetime.today()
                    age2 = today.year - birthdate2.year - ((today.month, today.day) < (birthdate2.month, birthdate2.day))
                    height2 = info2.loc[0, 'HEIGHT']
                    weight2 = info2.loc[0, 'WEIGHT']
                    position2 = info2.loc[0, 'POSITION']
                    college2 = info2.loc[0, 'SCHOOL']
                    teamID2 = info2.loc[0, 'TEAM_ID']
                    nbateam2 = info2.loc[0, 'TEAM_NAME']
                    headshot_url2 = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{other_player['id']}.png"
                    team_logo_url2 = f"https://cdn.nba.com/logos/nba/{teamID2}/global/L/logo.svg"

                    st.subheader("Player Info")
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(headshot_url2, caption=other_player['full_name'], width=200)
                    with col2:
                        st.markdown(f"**Age:** {age2}")
                        st.markdown(f"**Height:** {height2}")
                        st.markdown(f"**Weight:** {weight2} lbs")
                        st.markdown(f"**Position:** {position2}")
                        st.markdown(f"**College:** {college2}")
                        if college:
                            college_logo_url = college_logos.get(college2)
                            col_logo1, col_logo2 = st.columns([1, 5])
                            with col_logo1:
                                if college_logo_url:
                                    st.image(college_logo_url, width=100)
                                else:
                                    st.write("No logo available")
                            with col_logo2:
                                st.image(team_logo_url2, width=100)

                    career1 = playercareerstats.PlayerCareerStats(player_id=player['id'], per_mode36='PerGame').get_data_frames()[0]
                    time.sleep(1)
                    career2 = playercareerstats.PlayerCareerStats(player_id=other_player['id'], per_mode36='PerGame').get_data_frames()[0]
                    

                    career1_adv = compute_full_advanced_stats(career1)
                    career2_adv = compute_full_advanced_stats(career2)

                    excluded_cols = ['SEASON_ID', 'PLAYER_ID', 'TEAM_ID', 'LEAGUE_ID', 'TEAM_ABBREVIATION']

                    # Compute intersection of both player's stat columns
                    shared_stats = [col for col in career1_adv.columns
                                    if col in career2_adv.columns and col not in excluded_cols]

                    # Optional: Sort the list
                    shared_stats.sort()

                    available_stats = [col for col in career1.columns if col not in ["SEASON_ID", "SEASON_START", "TEAM_ID", "TEAM_ABBREVIATION", "PLAYER_ID"] and pd.api.types.is_numeric_dtype(career1[col])]
    
                    stat_choice = st.selectbox("📊 Choose a stat to compare: ", sorted(shared_stats))

                    career1_adv['SEASON_START'] = career1_adv['SEASON_ID'].str[:4].astype(int)
                    career2_adv['SEASON_START'] = career2_adv['SEASON_ID'].str[:4].astype(int)

                    common = career1_adv.merge(
                        career2_adv,
                        on='SEASON_START',
                        suffixes=(f"_{player['full_name']}", f"_{other_player['full_name']}")
                    )

                    if common.empty:
                        st.write("No overlapping seasons to compare.")
                    else:
                        x = common["SEASON_ID_" + player['full_name']]
                        y1 = common[stat_choice + "_" + player['full_name']]
                        y2 = common[stat_choice + "_" + other_player['full_name']]

                        fig, ax = plt.subplots()
                        ax.plot(x, y1, marker='o', label=player['full_name'])
                        ax.plot(x, y2, marker='o', label=other_player['full_name'])
                        ax.set_title(f"{stat_choice} Per Game — Overlapping Seasons")
                        ax.set_xlabel("Season")
                        ax.set_ylabel(f"{stat_choice} Per Game")
                        ax.legend()
                        plt.xticks(rotation=45)
                        st.pyplot(fig)

                        career1 = compute_full_advanced_stats(career1)
                        career2 = compute_full_advanced_stats(career2)

                        compare_cols = ['SEASON_ID', 'PTS', 'REB', 'AST', 'TS%', 'EFG%', 'PER', 'AST/TO', 'USG% (est)', 'PPS',]
                        c1 = career1[compare_cols].rename(columns={col: col + f" ({player['full_name']})" for col in compare_cols if col != 'SEASON_ID'})
                        c2 = career2[compare_cols].rename(columns={col: col + f" ({other_player['full_name']})" for col in compare_cols if col != 'SEASON_ID'})

                        career1_adv = career1_adv.drop(columns=['FG_PCT', 'FG3_PCT', 'FT_PCT'], errors='ignore')
                        career2_adv = career2_adv.drop(columns=['FG_PCT', 'FG3_PCT', 'FT_PCT'], errors='ignore')


                        merged_stats = pd.merge(c1, c2, on='SEASON_ID')

                        st.subheader("📊 Advanced Stats Comparison")
                        # Create two columns for side-by-side display
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown(f"**{player['full_name']}**")
                            st.dataframe(career1_adv.round(2), use_container_width=True)


                        with col2:
                            st.markdown(f"**{other_player['full_name']}**")
                            st.dataframe(career2_adv.round(2), use_container_width=True)

                        with st.expander("🧠 Ask the AI Assistant about these players"):
                            user_question = st.text_input("Ask something about these players:", key="ai_compare_question")
                            if user_question:
                                # Generate summaries
                                summary1 = generate_player_summary(player['full_name'], career1_adv, career1_adv)
                                summary2 = generate_player_summary(other_player['full_name'], career2_adv, career2_adv)

                                # Build full prompt
                                full_prompt = f"You are an expert NBA analyst. Here are the stat summaries for two players:\n\nPlayer 1: {player['full_name']}\n{summary1}\n\nPlayer 2: {other_player['full_name']}\n{summary2}\n\nQuestion: {user_question}\n\nAnswer like you're explaining to a smart basketball fan:"

                                # Call Gemini
                                with st.spinner("Analyzing..."):
                                    response = model.generate_content(full_prompt, generation_config={"max_output_tokens": 2048, "temperature": 0.7})
                                    st.markdown("### 🧠 AI Analysis")
                                    st.write(response.text)
            else:
                st.write("No second player found with that name.")
