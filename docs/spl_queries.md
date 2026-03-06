
# Splunk SPL Queries Used for Dataset Generation

These queries were used to extract telemetry associated with ATT&CK techniques during the Atomic Red Team simulations.

## Process Discovery (T1057)

index=wineventlog EventCode=4688
| xmlkv
| where CommandLine LIKE "%tasklist%" OR CommandLine LIKE "%wmic process%"
| eval Technique="T1057 - Process Discovery"
| table _time host user CommandLine ParentProcessName

## Network Connection Discovery (T1049)

index=wineventlog EventCode=4104
| search ScriptBlockText="Get-NetTCPConnection"
| append [
 search index=wineventlog EventCode=4688
 | xmlkv
 | search NewProcessName="*netstat.exe*"
]
| eval Technique="T1049 - Network Connection Discovery"
| table _time host user ScriptBlockText CommandLine

## File and Directory Discovery (T1083)

index=wineventlog EventCode=4104
| search ScriptBlockText="Get-ChildItem"
| search ScriptBlockText="-Recurse"
| eval Technique="T1083 - File Discovery"
| table _time host user ScriptBlockText

## Environment Variable Discovery (T1082)

index=wineventlog EventCode=4104
| search ScriptBlockText="Env:"
| eval Technique="T1082 - Environment Discovery"
| table _time host user ScriptBlockText

## Registry Discovery (T1012)

index=wineventlog (EventCode=4104 OR EventCode=4688)
| xmlkv
| search Command="*CurrentVersion\\Run*"
| eval Technique="T1012 - Registry Discovery"
| table _time host user Command

## Certutil Decode (T1140)

index=wineventlog EventCode=4688
| xmlkv
| search NewProcessName="*certutil.exe*"
| search CommandLine="*-decode*"
| eval Technique="T1140 - Decode Files"
| table _time host user CommandLine ParentProcessName
