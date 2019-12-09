function benchmark
{
    param([string]$program, [int]$threadCount = 0)
    $cmdOutput = Measure-Command {start-process ..\x64\Release\GaussianBlur.exe -ArgumentList "-p $program", "-t $threadCount", "-q" -Wait} | Out-String

    # The output ends with: 'TotalMilliseconds : 2009,4356'
    # Find this final by looking for the final whitespace
    $timeStartIdx = $cmdOutput.LastIndexOf(' ') + 1

    # Don't include the decimals after the comma
    $timeLength = $cmdOutput.LastIndexOf(',') - $timeStartIdx

    $time = $cmdOutput.Substring($timeStartIdx, $timeLength)

    return $time
}

# The amount of threads to run when benchmarking CUDA executions
$threadCounts = @(256, 1024, 2048, 4096, 8192, 69632)

# Write all test names (eg. 'CUDA (1024 threads)')
$csvOutput = "empty"
$csvOutput += ",CPU (1 thread)"

foreach ($threadCount in $threadCounts) {
    $csvOutput += ",CUDA ($threadCount threads)"
}

$csvOutput += "`n"

# Benchmark the CPU
Write-Output "CPU"
$time = benchmark -program "CPU"
Write-Output "Time: $time"
$csvOutput += ",$time"

foreach ($threadCount in $threadCounts) {
    Write-Output "CUDA ($threadCount threads)"
    $time = benchmark -program "CUDA" -threadCount $threadCount
    Write-Output "Time: $time"
    $csvOutput += ",$time"
}

Out-File -FilePath "benchmark.csv" -InputObject $csvOutput
Write-Output $csvOutput
