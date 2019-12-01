function benchmark
{
    param([string]$program, [int]$matrixWidthSize, [int]$threadCount = 0)
    $cmdOutput = Measure-Command {start-process ..\x64\Release\GaussianEliminationCUDA.exe -ArgumentList "-p $program", "-n $matrixWidth", "-t $threadCount", "-q" -Wait} | Out-String

    # The output ends with: 'TotalMilliseconds : 2009,4356'
    # Find this final by looking for the final whitespace
    $timeStartIdx = $cmdOutput.LastIndexOf(' ') + 1

    # Don't include the decimals after the comma
    $timeLength = $cmdOutput.LastIndexOf(',') - $timeStartIdx

    $time = $cmdOutput.Substring($timeStartIdx, $timeLength)

    return $time
}

# The amount of threads to run when benchmarking CUDA executions
$threadCounts = @(256, 1024, 2048, 65536)

$firstMatrixWidth = 4096
$lastMatrixWidth = 16384

# Amount of different matrix widths to perform benchmarks with
$matrixWidthSamples = 13
$matrixWidthIncrement = ($lastMatrixWidth - $firstMatrixWidth) / ($matrixWidthSamples-1)

# Write all test names (eg. 'CUDA (1024 threads)')
$csvOutput = "empty"
$csvOutput += ",CPU (1 thread)"

foreach ($threadCount in $threadCounts) {
    $csvOutput += ",CUDA ($threadCount threads)"
}

for ($matrixWidth = $firstMatrixWidth; $matrixWidth -le $lastMatrixWidth; $matrixWidth += $matrixWidthIncrement) {
    $csvOutput += "`n$matrixWidth"
    Write-Output "Matrix width: $matrixWidth"

    # Benchmark the CPU
   Write-Output "CPU"
   $time = benchmark -program "CPU" -matrixWidth $matrixWidth
   Write-Output "Time: $time"
   $csvOutput += ",$time"

    foreach ($threadCount in $threadCounts) {
        Write-Output "CUDA ($threadCount threads)"
        $time = benchmark -program "CUDA" -matrixWidth $matrixWidth -threadCount $threadCount
        Write-Output "Time: $time"
        $csvOutput += ",$time"
    }
}

Out-File -FilePath "benchmark.csv" -InputObject $csvOutput
Write-Output $csvOutput
