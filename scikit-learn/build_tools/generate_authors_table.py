"""
This script generates an html table of contributors, with names and avatars.
The list is generated from scikit-learn's teams on GitHub, plus a small number
of hard-coded contributors.

The table should be updated for each new inclusion in the teams.
Generating the table requires admin rights.
"""
import sys
import requests
import getpass
import time
from pathlib import Path
from os import path

print("user:", file=sys.stderr)
user = input()
token = getpass.getpass("access token:\n")
auth = (user, token)

LOGO_URL = "https://avatars2.githubusercontent.com/u/365630?v=4"
REPO_FOLDER = Path(path.abspath(__file__)).parent.parent


def get(url):
    for sleep_time in [10, 30, 0]:
        reply = requests.get(url, auth=auth)
        api_limit = (
            "message" in reply.json()
            and "API rate limit exceeded" in reply.json()["message"]
        )
        if not api_limit:
            break
        print("API rate limit exceeded, waiting..")
        time.sleep(sleep_time)

    reply.raise_for_status()
    return reply


def get_contributors():
    """Get the list of contributor profiles. Require admin rights."""
    # get core devs and triage team
    core_devs = []
    triage_team = []
    comm_team = []
    core_devs_id = 11523
    triage_team_id = 3593183
    comm_team_id = 5368696
    for team_id, lst in zip(
        (core_devs_id, triage_team_id, comm_team_id),
        (core_devs, triage_team, comm_team),
    ):
        for page in [1, 2]:  # 30 per page
            reply = get(f"https://api.github.com/teams/{team_id}/members?page={page}")
            lst.extend(reply.json())

    # get members of scikit-learn on GitHub
    members = []
    for page in [1, 2]:  # 30 per page
        reply = get(
            "https://api.github.com/orgs/scikit-learn/members?page=%d" % (page,)
        )
        members.extend(reply.json())

    # keep only the logins
    core_devs = set(c["login"] for c in core_devs)
    triage_team = set(c["login"] for c in triage_team)
    comm_team = set(c["login"] for c in comm_team)
    members = set(c["login"] for c in members)

    # add missing contributors with GitHub accounts
    members |= {"dubourg", "mbrucher", "thouis", "jarrodmillman"}
    # add missing contributors without GitHub accounts
    members |= {"Angel Soler Gollonet"}
    # remove CI bots
    members -= {"sklearn-ci", "sklearn-lgtm", "sklearn-wheels"}
    triage_team -= core_devs  # remove ogrisel from triage_team

    emeritus = members - core_devs - triage_team

    # get profiles from GitHub
    core_devs = [get_profile(login) for login in core_devs]
    emeritus = [get_profile(login) for login in emeritus]
    triage_team = [get_profile(login) for login in triage_team]
    comm_team = [get_profile(login) for login in comm_team]

    # sort by last name
    core_devs = sorted(core_devs, key=key)
    emeritus = sorted(emeritus, key=key)
    triage_team = sorted(triage_team, key=key)
    comm_team = sorted(comm_team, key=key)

    return core_devs, emeritus, triage_team, comm_team


def get_profile(login):
    """Get the GitHub profile from login"""
    print("get profile for %s" % (login,))
    try:
        profile = get("https://api.github.com/users/%s" % login).json()
    except requests.exceptions.HTTPError:
        return dict(name=login, avatar_url=LOGO_URL, html_url="")

    if profile["name"] is None:
        profile["name"] = profile["login"]

    # fix missing names
    missing_names = {
        "bthirion": "Bertrand Thirion",
        "dubourg": "Vincent Dubourg",
        "Duchesnay": "Edouard Duchesnay",
        "Lars": "Lars Buitinck",
        "MechCoder": "Manoj Kumar",
    }
    if profile["name"] in missing_names:
        profile["name"] = missing_names[profile["name"]]

    return profile


def key(profile):
    """Get a sorting key based on the lower case last name, then firstname"""
    components = profile["name"].lower().split(" ")
    return " ".join([components[-1]] + components[:-1])


def generate_table(contributors):
    lines = [
        ".. raw :: html\n",
        "    <!-- Generated by generate_authors_table.py -->",
        '    <div class="sk-authors-container">',
        "    <style>",
        "      img.avatar {border-radius: 10px;}",
        "    </style>",
    ]
    for contributor in contributors:
        lines.append("    <div>")
        lines.append(
            "    <a href='%s'><img src='%s' class='avatar' /></a> <br />"
            % (contributor["html_url"], contributor["avatar_url"])
        )
        lines.append("    <p>%s</p>" % (contributor["name"],))
        lines.append("    </div>")
    lines.append("    </div>")
    return "\n".join(lines)


def generate_list(contributors):
    lines = []
    for contributor in contributors:
        lines.append("- %s" % (contributor["name"],))
    return "\n".join(lines)


if __name__ == "__main__":

    core_devs, emeritus, triage_team, comm_team = get_contributors()

    with open(REPO_FOLDER / "doc" / "authors.rst", "w+") as rst_file:
        rst_file.write(generate_table(core_devs))

    with open(REPO_FOLDER / "doc" / "authors_emeritus.rst", "w+") as rst_file:
        rst_file.write(generate_list(emeritus))

    with open(REPO_FOLDER / "doc" / "triage_team.rst", "w+") as rst_file:
        rst_file.write(generate_table(triage_team))

    with open(REPO_FOLDER / "doc" / "communication_team.rst", "w+") as rst_file:
        rst_file.write(generate_table(comm_team))