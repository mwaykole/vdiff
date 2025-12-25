{{/*
Expand the name of the chart.
*/}}
{{- define "vdiff.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "vdiff.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "vdiff.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "vdiff.labels" -}}
helm.sh/chart: {{ include "vdiff.chart" . }}
{{ include "vdiff.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "vdiff.selectorLabels" -}}
app.kubernetes.io/name: {{ include "vdiff.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "vdiff.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "vdiff.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Model path - either from PVC or existing claim
*/}}
{{- define "vdiff.modelPath" -}}
{{- if .Values.model.storage.existingClaim }}
{{- .Values.model.storage.mountPath }}
{{- else if .Values.model.storage.pvc.enabled }}
{{- .Values.model.storage.mountPath }}
{{- else }}
{{- .Values.model.name }}
{{- end }}
{{- end }}

